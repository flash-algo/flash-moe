import torch
import triton
from triton import language as tl

from flash_moe.ops.utils import ensure_contiguous


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=8, num_stages=4),
    ],
    key=["hidden_size"],
)
@triton.heuristics(
    {
        "EVEN_N": lambda args: args["hidden_size"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _fwd_scores_kernel(
    X,
    W,
    S,
    sorted_token_ids,
    expert_offsets,
    stride_xm,
    stride_xn,
    stride_wk,
    stride_wn,
    stride_im,
    num_tokens,
    hidden_size: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Initialize offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_nb = tl.arange(0, BLOCK_N)

    # Load segment boundaries
    seg_start = tl.load(expert_offsets + pid_k)
    seg_end = tl.load(expert_offsets + pid_k + 1)

    # Compute pair ids
    pair_ids = seg_start + offs_m
    mask_m = pair_ids < seg_end

    # Load token ids
    token_ids = tl.load(
        sorted_token_ids + pair_ids * stride_im,
        mask=mask_m,
        other=0,
    )
    mask_m &= token_ids < num_tokens

    # Initialize pointers
    x_ptrs = X + token_ids[:, None] * stride_xm
    w_ptrs = W + pid_k * stride_wk

    # Initialize accumulator for s
    acc_s = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Loop over hidden dimension
    for start_n in range(0, hidden_size, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_nb

        # Load x
        if EVEN_N:
            x = tl.load(
                x_ptrs + offs_n[None, :] * stride_xn,
                mask=mask_m[:, None],
                other=0.0,
            )
        else:
            x = tl.load(
                x_ptrs + offs_n[None, :] * stride_xn,
                mask=mask_m[:, None] & (offs_n[None, :] < hidden_size),
                other=0.0,
            )

        # Load w
        if EVEN_N:
            w = tl.load(w_ptrs + offs_n * stride_wn)
        else:
            w = tl.load(
                w_ptrs + offs_n * stride_wn,
                mask=offs_n < hidden_size,
                other=0.0,
            )

        # Compute s
        acc_s += tl.sum(x * w[None, :], axis=1)

    # Write back s
    tl.store(
        S + pair_ids,
        acc_s,
        mask=mask_m,
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=8, num_stages=4),
    ],
    key=["hidden_size"],
    reset_to_zero=["Out"],
)
@triton.heuristics(
    {
        "EVEN_N": lambda args: args["hidden_size"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _fwd_states_kernel(
    S,
    W,
    G,
    Out,
    sorted_token_ids,
    expert_offsets,
    stride_sm,
    stride_wk,
    stride_wn,
    stride_gm,
    stride_im,
    stride_om,
    stride_on,
    num_tokens,
    hidden_size: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Initialize offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load segment boundaries
    seg_start = tl.load(expert_offsets + pid_k)
    seg_end = tl.load(expert_offsets + pid_k + 1)

    # Compute pair ids
    pair_ids = seg_start + offs_m
    mask_m = pair_ids < seg_end

    # Load token ids
    token_ids = tl.load(
        sorted_token_ids + pair_ids * stride_im,
        mask=mask_m,
        other=0,
    )
    mask_m &= token_ids < num_tokens

    # Initialize pointers
    s_ptrs = S + pair_ids * stride_sm
    g_ptrs = G + pair_ids * stride_gm
    w_ptrs = W + pid_k * stride_wk + offs_n * stride_wn
    o_ptrs = Out + token_ids[:, None] * stride_om + offs_n[None, :] * stride_on

    # Load s
    s = tl.load(s_ptrs, mask=mask_m, other=0.0)
    # Load g
    g = tl.load(g_ptrs, mask=mask_m, other=0.0)

    # Compute gated s
    gs = silu(s).cast(g.dtype) * g

    # Load w
    if EVEN_N:
        w = tl.load(w_ptrs)
    else:
        w = tl.load(w_ptrs, mask=offs_n < hidden_size, other=0.0)

    # Compute o
    o = gs[:, None] * w[None, :]

    # Write back o
    if EVEN_N:
        tl.atomic_add(o_ptrs, o, mask=mask_m[:, None])
    else:
        tl.atomic_add(
            o_ptrs,
            o,
            mask=mask_m[:, None] & (offs_n[None, :] < hidden_size),
        )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=8, num_stages=4),
    ],
    key=["hidden_size"],
)
@triton.heuristics(
    {
        "EVEN_N": lambda args: args["hidden_size"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _bwd_states_kernel(
    DO,
    W,
    G,
    S,
    DG,
    DS,
    sorted_token_ids,
    sorted_pair_ids,
    expert_offsets,
    stride_dom,
    stride_don,
    stride_wk,
    stride_wn,
    stride_gm,
    stride_sm,
    stride_dgm,
    stride_dsm,
    stride_im,
    stride_pm,
    stride_off,
    num_tokens,
    hidden_size: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Initialize offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_nb = tl.arange(0, BLOCK_N)

    # Load segment boundaries
    seg_start = tl.load(expert_offsets + (pid_k) * stride_off)
    seg_end = tl.load(expert_offsets + (pid_k + 1) * stride_off)

    # Compute pair ids
    pair_ids = seg_start + offs_m
    mask_m = pair_ids < seg_end

    # Load token ids
    token_ids = tl.load(
        sorted_token_ids + pair_ids * stride_im,
        mask=mask_m,
        other=0,
    )
    mask_m &= token_ids < num_tokens

    # Initialize pointers
    g_ptrs = G + pair_ids * stride_gm
    s_ptrs = S + pair_ids * stride_sm
    do_ptrs = DO + token_ids[:, None] * stride_dom
    w_ptrs = W + pid_k * stride_wk
    p_ptrs = sorted_pair_ids + pair_ids * stride_pm

    # Load p
    p = tl.load(p_ptrs, mask=mask_m, other=0)

    # Load g
    g = tl.load(g_ptrs, mask=mask_m, other=0.0)

    # Load s
    s = tl.load(s_ptrs, mask=mask_m, other=0.0).to(tl.float32)

    # Initialize accumulator for dg
    acc_dg = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Loop over hidden dimension
    for start_n in range(0, hidden_size, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_nb

        # Load do
        if EVEN_N:
            do = tl.load(
                do_ptrs + offs_n[None, :] * stride_don,
                mask=mask_m[:, None],
                other=0.0,
            )
        else:
            do = tl.load(
                do_ptrs + offs_n[None, :] * stride_don,
                mask=mask_m[:, None] & (offs_n[None, :] < hidden_size),
                other=0.0,
            )

        # Load w
        if EVEN_N:
            w = tl.load(w_ptrs + offs_n * stride_wn)
        else:
            w = tl.load(
                w_ptrs + offs_n * stride_wn,
                mask=offs_n < hidden_size,
                other=0.0,
            )

        # Compute dg
        acc_dg += tl.sum(do * w[None, :], axis=1)

    act_s = tl.sigmoid(s)

    # Compute dg
    dg = (acc_dg * act_s * s).to(g.dtype)

    # Write back dg
    tl.store(DG + p * stride_dgm, dg, mask=mask_m)

    # Compute dsilu
    dsilu = act_s + s * act_s * (1.0 - act_s)

    # Compute ds
    ds = acc_dg * g.to(tl.float32) * dsilu

    # Write back ds
    tl.store(DS + pair_ids * stride_dsm, ds, mask=mask_m)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=8, num_stages=4),
    ],
    key=["hidden_size"],
    reset_to_zero=["DX", "DWD", "DWU"],
)
@triton.heuristics(
    {
        "EVEN_N": lambda args: args["hidden_size"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _bwd_ec_scores_kernel(
    DS,
    X,
    W,
    DO,
    G,
    S,
    DX,
    DWD,
    DWU,
    sorted_token_ids,
    expert_offsets,
    stride_dsm,
    stride_xm,
    stride_xn,
    stride_wk,
    stride_wn,
    stride_dom,
    stride_don,
    stride_gm,
    stride_sm,
    stride_dxm,
    stride_dxn,
    stride_dwdk,
    stride_dwdn,
    stride_dwuk,
    stride_dwun,
    stride_im,
    stride_off,
    num_tokens,
    hidden_size: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Initialize offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load segment boundaries
    seg_start = tl.load(expert_offsets + (pid_k) * stride_off)
    seg_end = tl.load(expert_offsets + (pid_k + 1) * stride_off)

    # Compute pair ids
    pair_ids = seg_start + offs_m
    mask_m = pair_ids < seg_end

    # Load token ids
    token_ids = tl.load(
        sorted_token_ids + pair_ids * stride_im,
        mask=mask_m,
        other=0,
    )
    mask_m &= token_ids < num_tokens

    # Initialize pointers
    ds_ptrs = DS + pair_ids * stride_dsm
    g_ptrs = G + pair_ids * stride_gm
    s_ptrs = S + pair_ids * stride_sm
    w_ptrs = W + pid_k * stride_wk + offs_n * stride_wn
    dx_ptrs = DX + token_ids[:, None] * stride_dxm + offs_n[None, :] * stride_dxn
    x_ptrs = X + token_ids[:, None] * stride_xm + offs_n[None, :] * stride_xn
    dwd_ptrs = DWD + pid_k * stride_dwdk + offs_n * stride_dwdn
    do_ptrs = DO + token_ids[:, None] * stride_dom + offs_n[None, :] * stride_don
    dwu_ptrs = DWU + pid_k * stride_dwuk + offs_n * stride_dwun

    # Load ds
    ds = tl.load(ds_ptrs, mask=mask_m, other=0.0)

    # Load w
    if EVEN_N:
        w = tl.load(w_ptrs)
    else:
        w = tl.load(
            w_ptrs,
            mask=offs_n < hidden_size,
            other=0.0,
        )

    # Compute dx
    dx = ds[:, None] * w[None, :]

    # Write back dx
    if EVEN_N:
        tl.atomic_add(dx_ptrs, dx.to(w.dtype), mask=mask_m[:, None])
    else:
        tl.atomic_add(
            dx_ptrs,
            dx.to(w.dtype),
            mask=mask_m[:, None] & (offs_n[None, :] < hidden_size),
        )

    # Load x
    if EVEN_N:
        x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
    else:
        x = tl.load(
            x_ptrs,
            mask=mask_m[:, None] & (offs_n[None, :] < hidden_size),
            other=0.0,
        )

    # Compute dwd
    dwd = tl.sum(ds[:, None] * x, axis=0)

    # Write back dwd
    if EVEN_N:
        tl.atomic_add(dwd_ptrs, dwd.to(x.dtype))
    else:
        tl.atomic_add(
            dwd_ptrs,
            dwd.to(x.dtype),
            mask=offs_n[None, :] < hidden_size,
        )

    # Load g
    g = tl.load(g_ptrs, mask=mask_m, other=0.0)

    # Load s
    s = tl.load(s_ptrs, mask=mask_m, other=0.0)

    gated = silu(s) * g

    # Load do
    if EVEN_N:
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    else:
        do = tl.load(
            do_ptrs,
            mask=mask_m[:, None] & (offs_n[None, :] < hidden_size),
            other=0.0,
        )

    # Compute dwu
    dwu = tl.sum(gated[:, None] * do, axis=0)

    # Write back dwu
    if EVEN_N:
        tl.atomic_add(dwu_ptrs, dwu.to(x.dtype))
    else:
        tl.atomic_add(
            dwu_ptrs,
            dwu.to(x.dtype),
            mask=offs_n[None, :] < hidden_size,
        )


def _preprocess_expert_centric_indices_pytorch(
    Indices: torch.Tensor,
    G: torch.Tensor,
    num_experts: int,
):
    """
    Convert token-centric token-expert pairs to expert-centric format.

    Args:
        Indices: (batch_size * seq_len, num_experts_per_tok)
        G: (batch_size * seq_len, num_experts_per_tok)
        num_experts: total number of experts

    Returns:
        sorted_token_ids: (batch_size * seq_len * num_experts_per_tok)
        sorted_expert_ids: (batch_size * seq_len * num_experts_per_tok)
        sorted_weights: (batch_size * seq_len * num_experts_per_tok)
        expert_offsets: (num_experts + 1)
        max_pairs_per_expert: maximum number of (token, expert) pairs assigned to any expert
        sorted_pair_ids: mapping from sorted position -> original flattened (token, k) position
    """
    num_tokens, num_experts_per_tok = Indices.shape

    token_ids = torch.arange(
        num_tokens, dtype=torch.int64, device=Indices.device
    ).unsqueeze(1)
    token_ids = token_ids.expand(-1, num_experts_per_tok).reshape(-1)
    expert_ids = Indices.reshape(-1).to(torch.int64)
    G = G.reshape(-1)

    sorted_expert_ids, sorted_pair_ids = torch.sort(expert_ids, stable=True)
    sorted_token_ids = token_ids[sorted_pair_ids]
    sorted_G = G[sorted_pair_ids]

    expert_counts = torch.bincount(sorted_expert_ids, minlength=num_experts)
    expert_offsets = torch.zeros(
        num_experts + 1, dtype=torch.int64, device=Indices.device
    )
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

    max_pairs_per_expert = int(expert_counts.max().item()) if num_experts > 0 else 0

    return (
        sorted_token_ids,
        sorted_G,
        expert_offsets,
        max_pairs_per_expert,
        sorted_pair_ids,
    )


def _flash_expert_forward(
    X: torch.Tensor,
    W_d: torch.Tensor,
    W_u: torch.Tensor,
    Indices: torch.Tensor,
    G: torch.Tensor,
):
    num_tokens, hidden_size = X.shape
    num_experts = W_d.shape[0]

    (
        sorted_token_ids,
        sorted_G,
        expert_offsets,
        max_pairs_per_expert,
        sorted_pair_ids,
    ) = _preprocess_expert_centric_indices_pytorch(Indices, G, num_experts)

    # Scores are stored as float32 for backward
    sorted_S = torch.empty(
        (sorted_token_ids.numel(),),
        device=X.device,
        dtype=torch.float32,
    )

    # Accumulate per-token output via atomics
    Out = torch.zeros_like(X)

    def grid(META):
        return (num_experts, triton.cdiv(max_pairs_per_expert, META["BLOCK_M"]))

    _fwd_scores_kernel[grid](
        X,
        W_d,
        sorted_S,
        sorted_token_ids,
        expert_offsets,
        X.stride(0),
        X.stride(1),
        W_d.stride(0),
        W_d.stride(1),
        sorted_token_ids.stride(0),
        num_tokens=num_tokens,
        hidden_size=hidden_size,
    )

    def grid(META):
        return (
            num_experts,
            triton.cdiv(max_pairs_per_expert, META["BLOCK_M"]),
            triton.cdiv(hidden_size, META["BLOCK_N"]),
        )

    _fwd_states_kernel[grid](
        sorted_S,
        W_u,
        sorted_G,
        Out,
        sorted_token_ids,
        expert_offsets,
        sorted_S.stride(0),
        W_u.stride(0),
        W_u.stride(1),
        sorted_G.stride(0),
        sorted_token_ids.stride(0),
        Out.stride(0),
        Out.stride(1),
        num_tokens=num_tokens,
        hidden_size=hidden_size,
    )

    return Out, sorted_S, sorted_G, sorted_token_ids, expert_offsets, sorted_pair_ids


def _flash_expert_backward(
    dO: torch.Tensor,
    X: torch.Tensor,
    W_d: torch.Tensor,
    W_u: torch.Tensor,
    G: torch.Tensor,
    S: torch.Tensor,
    sorted_G: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_offsets: torch.Tensor,
    sorted_pair_ids: torch.Tensor,
):
    num_tokens, hidden_size = X.shape
    num_experts = W_d.shape[0]

    # recompute max_pairs_per_expert because it is not saved in ctx
    max_pairs_per_expert = int((expert_offsets[1:] - expert_offsets[:-1]).max().item())
    dX = torch.zeros_like(X)
    dW_d = torch.zeros_like(W_d)
    dW_u = torch.zeros_like(W_u)
    dG = torch.empty_like(G).view(-1)
    dS = torch.empty_like(S)

    def grid_states(META):
        return (num_experts, triton.cdiv(max_pairs_per_expert, META["BLOCK_M"]))

    _bwd_states_kernel[grid_states](
        dO,
        W_u,
        sorted_G,
        S,
        dG,
        dS,
        sorted_token_ids,
        sorted_pair_ids,
        expert_offsets,
        dO.stride(0),
        dO.stride(1),
        W_u.stride(0),
        W_u.stride(1),
        sorted_G.stride(0),
        S.stride(0),
        dG.stride(0),
        dS.stride(0),
        sorted_token_ids.stride(0),
        sorted_pair_ids.stride(0),
        expert_offsets.stride(0),
        num_tokens=num_tokens,
        hidden_size=hidden_size,
    )

    def grid_scores(META):
        return (
            num_experts,
            triton.cdiv(max_pairs_per_expert, META["BLOCK_M"]),
            triton.cdiv(hidden_size, META["BLOCK_N"]),
        )

    _bwd_ec_scores_kernel[grid_scores](
        dS,
        X,
        W_d,
        dO,
        sorted_G,
        S,
        dX,
        dW_d,
        dW_u,
        sorted_token_ids,
        expert_offsets,
        dS.stride(0),
        X.stride(0),
        X.stride(1),
        W_d.stride(0),
        W_d.stride(1),
        dO.stride(0),
        dO.stride(1),
        sorted_G.stride(0),
        S.stride(0),
        dX.stride(0),
        dX.stride(1),
        dW_d.stride(0),
        dW_d.stride(1),
        dW_u.stride(0),
        dW_u.stride(1),
        sorted_token_ids.stride(0),
        expert_offsets.stride(0),
        num_tokens=num_tokens,
        hidden_size=hidden_size,
    )

    return dX, dW_d, dW_u, dG.view_as(G)


class FlashExpertFunc(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, hidden_states, down_weights, up_weights, indices, routing_weights):
        """
        Expert-Centric forward pass for flash expert.

        This implementation reduces memory bandwidth by loading each expert's weights
        only once per forward pass, then processing all tokens assigned to that expert.

        Args:
            hidden_states: (batch_size * seq_len, hidden_size)
            down_weights: (num_experts, hidden_size)
            up_weights: (num_experts, hidden_size)
            indices: (batch_size * seq_len, num_experts_per_tok)
            routing_weights: (batch_size * seq_len, num_experts_per_tok)

        Returns:
            experts_states: (batch_size * seq_len, hidden_size)
        """

        (
            experts_states,
            expert_scores,
            sorted_routing_weights,
            sorted_token_ids,
            expert_offsets,
            sorted_pair_ids,
        ) = _flash_expert_forward(
            hidden_states,
            down_weights,
            up_weights,
            indices,
            routing_weights,
        )

        ctx.save_for_backward(
            hidden_states,
            down_weights,
            up_weights,
            routing_weights,
            expert_scores,
            sorted_routing_weights,
            sorted_token_ids,
            expert_offsets,
            sorted_pair_ids,
        )

        return experts_states

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dexperts_states):
        (
            hidden_states,
            down_weights,
            up_weights,
            routing_weights,
            expert_scores,
            sorted_routing_weights,
            sorted_token_ids,
            expert_offsets,
            sorted_pair_ids,
        ) = ctx.saved_tensors

        dhidden_states, ddown_weights, dup_weights, drouting_weights = (
            _flash_expert_backward(
                dexperts_states,
                hidden_states,
                down_weights,
                up_weights,
                routing_weights,
                expert_scores,
                sorted_routing_weights,
                sorted_token_ids,
                expert_offsets,
                sorted_pair_ids,
            )
        )

        return dhidden_states, ddown_weights, dup_weights, None, drouting_weights


def triton_flash_expert_func(
    hidden_states, down_weights, up_weights, indices, routing_weights
):
    return FlashExpertFunc.apply(
        hidden_states, down_weights, up_weights, indices, routing_weights
    )
