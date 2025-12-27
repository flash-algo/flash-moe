import math

import torch
import cuda.tile as ct

from flash_moe.ops.utils import ensure_contiguous


ConstInt = ct.Constant[int]


@ct.kernel
def _pkm_fwd_kernel(
    values,
    routing_weights,
    indices,
    output,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    topk: ConstInt,
):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    # Compute token indices for this tile
    token_indices = bid_m * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    col_indices = bid_n * TILE_N + ct.arange(TILE_N, dtype=ct.int32)

    # Initialize accumulator
    acc = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)

    # Loop over topk experts
    for k in range(topk):
        # Gather indices for this k: (TILE_M,)
        k_indices = ct.full((TILE_M,), k, dtype=ct.int32)
        expert_ids = ct.gather(indices, (token_indices, k_indices))

        # Gather routing weights for this k: (TILE_M,)
        weights = ct.gather(routing_weights, (token_indices, k_indices))
        weights = ct.astype(weights, ct.float32)

        # Gather values using expert indices: (TILE_M, TILE_N)
        values_tile = ct.gather(values, (expert_ids[:, None], col_indices[None, :]))
        values_tile = ct.astype(values_tile, ct.float32)

        # Accumulate weighted values
        acc = acc + weights[:, None] * values_tile

    # Store output
    ct.scatter(
        output,
        (token_indices[:, None], col_indices[None, :]),
        ct.astype(acc, output.dtype),
    )


def pkm_forward(
    values: torch.Tensor,
    routing_weights: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """
    PKM forward pass using cutile kernel.

    Args:
        values: Expert values, shape (num_experts, hidden_size)
        routing_weights: Router weights, shape (num_tokens, topk)
        indices: Expert indices, shape (num_tokens, topk)

    Returns:
        Output tensor, shape (num_tokens, hidden_size)
    """
    _, hidden_size = values.shape
    num_tokens, topk = indices.shape

    # Allocate output tensor
    output = torch.empty(
        (num_tokens, hidden_size),
        device=values.device,
        dtype=values.dtype,
    )

    # Tile sizes
    TILE_M = 64
    TILE_N = 64

    # Launch kernel
    grid = (
        math.ceil(num_tokens / TILE_M),
        math.ceil(hidden_size / TILE_N),
    )

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _pkm_fwd_kernel,
        (
            values,
            routing_weights,
            indices,
            output,
            TILE_M,
            TILE_N,
            topk,
        ),
    )

    return output


class CutilePKMFunc(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        values: torch.Tensor,
        routing_weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        output = pkm_forward(values, routing_weights, indices)
        ctx.save_for_backward(values, routing_weights, indices)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        raise NotImplementedError("CutilePKMFunc backward is not implemented yet.")


def cutile_pkm_func(
    values: torch.Tensor,
    routing_weights: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    return CutilePKMFunc.apply(values, routing_weights, indices)
