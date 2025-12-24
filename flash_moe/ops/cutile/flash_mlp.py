import torch
import cuda.tile as ct

from flash_moe.ops.utils import next_power_of_2, ensure_contiguous


ConstInt = ct.Constant[int]


def _sigmoid(x):
    denom = ct.add(1, ct.exp(-x), flush_to_zero=True)
    sigmoid = ct.truediv(
        1.0, denom, flush_to_zero=True, rounding_mode=ct.RoundingMode.APPROX
    )
    return sigmoid


@ct.kernel
def _swiglu_forward_kernel(a, b, c, TILE_N: ConstInt):
    bid = ct.bid(0)
    n_cols = a.shape[1]
    n_tiles_n = ct.cdiv(n_cols, TILE_N)
    bid_m = bid // n_tiles_n
    bid_n = bid % n_tiles_n
    col_start = bid_n * TILE_N

    zero_pad = ct.PaddingMode.ZERO
    # sigmoid requires type float32
    ta = ct.load(a, (bid_m, col_start), (1, TILE_N), padding_mode=zero_pad).astype(
        ct.float32
    )
    tb = ct.load(b, (bid_m, col_start), (1, TILE_N), padding_mode=zero_pad).astype(
        ct.float32
    )

    sigmoid = _sigmoid(ta)
    silu = ct.mul(ta, sigmoid, flush_to_zero=True)
    tc = ct.mul(silu, tb, flush_to_zero=True)

    ct.store(c, (bid_m, col_start), tc.astype(c.dtype))


@ct.kernel
def _swiglu_backward_kernel(dc, a, b, da, db, TILE_N: ConstInt):
    bid = ct.bid(0)
    n_cols = a.shape[1]
    n_tiles_n = ct.cdiv(n_cols, TILE_N)
    bid_m = bid // n_tiles_n
    bid_n = bid % n_tiles_n
    col_start = bid_n * TILE_N

    zero_pad = ct.PaddingMode.ZERO
    # sigmoid requires type float32
    tdc = ct.load(dc, (bid_m, col_start), (1, TILE_N), padding_mode=zero_pad).astype(
        ct.float32
    )
    ta = ct.load(a, (bid_m, col_start), (1, TILE_N), padding_mode=zero_pad).astype(
        ct.float32
    )
    tb = ct.load(b, (bid_m, col_start), (1, TILE_N), padding_mode=zero_pad).astype(
        ct.float32
    )

    # recomputation to save memory
    sigmoid = _sigmoid(ta)
    silu = ct.mul(ta, sigmoid, flush_to_zero=True)

    tdb = ct.mul(tdc, silu, flush_to_zero=True)
    one_minus = ct.sub(1.0, sigmoid, flush_to_zero=True)
    term = ct.add(
        ct.mul(silu, one_minus, flush_to_zero=True), sigmoid, flush_to_zero=True
    )
    tda = ct.mul(ct.mul(tdc, term, flush_to_zero=True), tb, flush_to_zero=True)

    ct.store(da, (bid_m, col_start), tda.astype(da.dtype))
    ct.store(db, (bid_m, col_start), tdb.astype(db.dtype))


def swiglu_forward(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ori_shape = a.shape

    n_cols = ori_shape[-1]
    a = a.reshape(-1, n_cols).contiguous()
    b = b.reshape(-1, n_cols).contiguous()
    c = torch.empty_like(a)

    tile_n = 1024 if n_cols >= 1024 else next_power_of_2(n_cols)
    n_tiles_n = (n_cols + tile_n - 1) // tile_n
    grid = (a.shape[0] * n_tiles_n,)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _swiglu_forward_kernel,
        (a, b, c, tile_n),
    )
    return a, b, c.view(*ori_shape)


def swiglu_backward(
    a: torch.Tensor, b: torch.Tensor, dc: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.reshape(-1, n_cols).contiguous()
    a = a.reshape(-1, n_cols).contiguous()
    b = b.reshape(-1, n_cols).contiguous()
    da = torch.empty_like(a)
    db = torch.empty_like(b)

    tile_n = 1024 if n_cols >= 1024 else next_power_of_2(n_cols)
    n_tiles_n = (n_cols + tile_n - 1) // tile_n
    grid = (dc.shape[0] * n_tiles_n,)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _swiglu_backward_kernel,
        (dc, a, b, da, db, tile_n),
    )
    return da.view(*ori_shape), db.view(*ori_shape)


class CutileSwiGLUFunc(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a, b, c = swiglu_forward(a, b)
        ctx.save_for_backward(a, b)
        ctx.ori_shape = a.shape
        return c

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dc: torch.Tensor):
        a, b = ctx.saved_tensors
        da, db = swiglu_backward(a.view(ctx.ori_shape), b.view(ctx.ori_shape), dc)
        return da, db


def cutile_flash_mlp_func(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    return torch.matmul(
        CutileSwiGLUFunc.apply(
            torch.matmul(x, gate_weight.t()),
            torch.matmul(x, up_weight.t()),
        ),
        down_weight.t(),
    )
