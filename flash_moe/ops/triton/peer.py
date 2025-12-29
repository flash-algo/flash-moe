import torch
import triton
from triton import language as tl

from flash_moe.ops.utils import ensure_contiguous, next_power_of_2


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128}, num_warps=8, num_stages=2),
    ],
    key=["num_tokens", "hidden_size", "topk"],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["num_tokens"] % args["BLOCK_M"] == 0,
    }
)
@triton.jit
def _peer_fwd_kernel(
    HIDDEN_STATES,
    DOWN_EMBED,
    UP_EMBED,
    ROUTING_WEIGHTS,
    INDICES,
    OUTPUT,
    stride_hm,
    stride_hn,
    stride_de,
    stride_dn,
    stride_ue,
    stride_un,
    stride_rm,
    stride_rk,
    stride_im,
    stride_ik,
    stride_om,
    stride_on,
    num_tokens,
    hidden_size,
    topk,
    HIDDEN_SIZE_PAD: tl.constexpr,
    EVEN_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)

    # Token indices for this block: (BLOCK_M,)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < num_tokens

    # Hidden dimension indices: (HIDDEN_SIZE_PAD,)
    offs_n = tl.arange(0, HIDDEN_SIZE_PAD)
    mask_n = offs_n < hidden_size

    # Load and cache hidden_states for all tokens in this block: (BLOCK_M, HIDDEN_SIZE_PAD)
    hidden_ptrs = (
        HIDDEN_STATES + offs_m[:, None] * stride_hm + offs_n[None, :] * stride_hn
    )
    if EVEN_M:
        hidden_cached = tl.load(hidden_ptrs, mask=mask_n[None, :], other=0.0)
    else:
        hidden_cached = tl.load(
            hidden_ptrs,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        )
    hidden_cached = hidden_cached.to(tl.float32)

    # Initialize output accumulator: (BLOCK_M, HIDDEN_SIZE_PAD)
    acc = tl.zeros((BLOCK_M, HIDDEN_SIZE_PAD), dtype=tl.float32)

    # Loop over topk experts
    for k in range(topk):
        # Get expert index for this k: (BLOCK_M,)
        if EVEN_M:
            expert_idx = tl.load(INDICES + offs_m * stride_im + k * stride_ik)
        else:
            expert_idx = tl.load(
                INDICES + offs_m * stride_im + k * stride_ik,
                mask=mask_m,
                other=0,
            )

        # Get routing weight for this k: (BLOCK_M,)
        if EVEN_M:
            routing_weight = tl.load(
                ROUTING_WEIGHTS + offs_m * stride_rm + k * stride_rk
            )
        else:
            routing_weight = tl.load(
                ROUTING_WEIGHTS + offs_m * stride_rm + k * stride_rk,
                mask=mask_m,
                other=0.0,
            )

        # Load down_embed for this expert: (BLOCK_M, HIDDEN_SIZE_PAD)
        down_ptrs = (
            DOWN_EMBED + expert_idx[:, None] * stride_de + offs_n[None, :] * stride_dn
        )
        if EVEN_M:
            down = tl.load(down_ptrs, mask=mask_n[None, :], other=0.0)
        else:
            down = tl.load(
                down_ptrs,
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            )

        # Compute dot product using cached hidden_states: (BLOCK_M,)
        dot_product = tl.sum(hidden_cached * down.to(tl.float32), axis=1)

        # Apply GELU activation: gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
        INV_SQRT_2 = 0.7071067811865476
        gelu_out = 0.5 * dot_product * (1.0 + tl.math.erf(dot_product * INV_SQRT_2))

        # Multiply by routing weight: (BLOCK_M,)
        score = gelu_out * routing_weight.to(tl.float32)

        # Load up_embed for this expert: (BLOCK_M, HIDDEN_SIZE_PAD)
        up_ptrs = (
            UP_EMBED + expert_idx[:, None] * stride_ue + offs_n[None, :] * stride_un
        )
        if EVEN_M:
            up = tl.load(up_ptrs, mask=mask_n[None, :], other=0.0)
        else:
            up = tl.load(
                up_ptrs,
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            )

        # Accumulate: score[:, None] * up -> (BLOCK_M, HIDDEN_SIZE_PAD)
        acc += score[:, None] * up.to(tl.float32)

    # Store output: (BLOCK_M, HIDDEN_SIZE_PAD)
    output_ptrs = OUTPUT + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    if EVEN_M:
        tl.store(output_ptrs, acc.to(OUTPUT.dtype.element_ty), mask=mask_n[None, :])
    else:
        tl.store(
            output_ptrs,
            acc.to(OUTPUT.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


def peer_forward(
    hidden_states: torch.Tensor,
    down_embed: torch.Tensor,
    up_embed: torch.Tensor,
    routing_weights: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """
    PEER forward pass using Triton kernel.

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

    # Pad hidden_size to next power of 2 for efficient Triton kernel
    hidden_size_pad = next_power_of_2(hidden_size)

    # Allocate output tensor
    output = torch.empty(
        (num_tokens, hidden_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    # Launch kernel with 1D grid over tokens
    def grid(META):
        return (triton.cdiv(num_tokens, META["BLOCK_M"]),)

    _peer_fwd_kernel[grid](
        hidden_states,
        down_embed,
        up_embed,
        routing_weights,
        indices,
        output,
        hidden_states.stride(0),
        hidden_states.stride(1),
        down_embed.stride(0),
        down_embed.stride(1),
        up_embed.stride(0),
        up_embed.stride(1),
        routing_weights.stride(0),
        routing_weights.stride(1),
        indices.stride(0),
        indices.stride(1),
        output.stride(0),
        output.stride(1),
        num_tokens,
        hidden_size,
        topk,
        hidden_size_pad,
    )

    return output


class TritonPEERFunc(torch.autograd.Function):
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
        raise NotImplementedError("TritonPEERFunc backward is not implemented yet.")


def triton_peer_func(
    hidden_states: torch.Tensor,
    down_embed: torch.Tensor,
    up_embed: torch.Tensor,
    routing_weights: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    return TritonPEERFunc.apply(
        hidden_states, down_embed, up_embed, routing_weights, indices
    )
