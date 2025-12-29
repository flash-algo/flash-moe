import math

import torch
import cuda.tile as ct

from flash_moe.ops.utils import ensure_contiguous


ConstInt = ct.Constant[int]


@ct.kernel
def _peer_fwd_kernel(
    hidden_states,
    down_embed,
    up_embed,
    routing_weights,
    indices,
    output,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    hidden_size: ConstInt,
    topk: ConstInt,
):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    # Compute token indices for this tile: (TILE_M,)
    token_indices = bid_m * TILE_M + ct.arange(TILE_M, dtype=ct.int32)

    # Output column indices for this block: (TILE_N,)
    out_col_indices = bid_n * TILE_N + ct.arange(TILE_N, dtype=ct.int32)

    # Initialize output accumulator: (TILE_M, TILE_N)
    acc = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)

    # Loop over topk experts
    for k in range(topk):
        # Gather expert indices for this k: (TILE_M,)
        k_indices = ct.full((TILE_M,), k, dtype=ct.int32)
        expert_ids = ct.gather(indices, (token_indices, k_indices))

        # Gather routing weights for this k: (TILE_M,)
        routing_weight = ct.gather(routing_weights, (token_indices, k_indices))
        routing_weight = ct.astype(routing_weight, ct.float32)

        # Compute dot product between hidden_states and down_embed
        # Accumulate across all hidden dimensions in tiles
        dot_product = ct.full((TILE_M,), 0.0, dtype=ct.float32)

        for n_start in range(0, hidden_size, TILE_N):
            n_indices = n_start + ct.arange(TILE_N, dtype=ct.int32)

            # Gather hidden_states[token_indices, n_indices]: (TILE_M, TILE_N)
            hidden_tile = ct.gather(
                hidden_states, (token_indices[:, None], n_indices[None, :])
            )
            hidden_tile = ct.astype(hidden_tile, ct.float32)

            # Gather down_embed[expert_ids, n_indices]: (TILE_M, TILE_N)
            down_tile = ct.gather(down_embed, (expert_ids[:, None], n_indices[None, :]))
            down_tile = ct.astype(down_tile, ct.float32)

            # Accumulate dot product
            dot_product = dot_product + ct.sum(hidden_tile * down_tile, axis=1)

        # Apply GELU activation: gelu(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        SQRT_2_OVER_PI = 0.7978845608028654
        x_cubed = dot_product * dot_product * dot_product
        inner = SQRT_2_OVER_PI * (dot_product + 0.044715 * x_cubed)
        gelu_out = 0.5 * dot_product * (1.0 + ct.tanh(inner))

        # Multiply by routing weight: (TILE_M,)
        score = gelu_out * routing_weight

        # Gather up_embed for this expert at output columns: (TILE_M, TILE_N)
        up_tile = ct.gather(up_embed, (expert_ids[:, None], out_col_indices[None, :]))
        up_tile = ct.astype(up_tile, ct.float32)

        # Accumulate: score[:, None] * up_tile -> (TILE_M, TILE_N)
        acc = acc + score[:, None] * up_tile

    # Store output
    ct.scatter(
        output,
        (token_indices[:, None], out_col_indices[None, :]),
        ct.astype(acc, output.dtype),
    )


def peer_forward(
    hidden_states: torch.Tensor,
    down_embed: torch.Tensor,
    up_embed: torch.Tensor,
    routing_weights: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """
    PEER forward pass using cutile kernel.

    The PEER (Parameter Efficient Expert Retrieval) algorithm:
    1. Compute dot products between hidden_states and down_embed for each selected expert
    2. Apply GELU activation
    3. Multiply by routing weights
    4. Compute weighted sum of up_embed

    Args:
        hidden_states: Input tensor, shape (num_tokens, hidden_size)
        down_embed: Down projection embeddings, shape (num_experts, hidden_size)
        up_embed: Up projection embeddings, shape (num_experts, hidden_size)
        routing_weights: Router weights, shape (num_tokens, topk)
        indices: Expert indices, shape (num_tokens, topk)

    Returns:
        Output tensor, shape (num_tokens, hidden_size)
    """
    num_tokens, hidden_size = hidden_states.shape
    _, topk = indices.shape

    # Allocate output tensor
    output = torch.empty(
        (num_tokens, hidden_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    # Tile sizes
    TILE_M = 64
    TILE_N = 64

    # Launch kernel with 2D grid
    grid = (
        math.ceil(num_tokens / TILE_M),
        math.ceil(hidden_size / TILE_N),
    )

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _peer_fwd_kernel,
        (
            hidden_states,
            down_embed,
            up_embed,
            routing_weights,
            indices,
            output,
            TILE_M,
            TILE_N,
            hidden_size,
            topk,
        ),
    )

    return output


class CutilePEERFunc(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        down_embed: torch.Tensor,
        up_embed: torch.Tensor,
        routing_weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        output = peer_forward(
            hidden_states, down_embed, up_embed, routing_weights, indices
        )
        ctx.save_for_backward(
            hidden_states, down_embed, up_embed, routing_weights, indices
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        raise NotImplementedError("CutilePEERFunc backward is not implemented yet.")


def cutile_peer_func(
    hidden_states: torch.Tensor,
    down_embed: torch.Tensor,
    up_embed: torch.Tensor,
    routing_weights: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    return CutilePEERFunc.apply(
        hidden_states, down_embed, up_embed, routing_weights, indices
    )
