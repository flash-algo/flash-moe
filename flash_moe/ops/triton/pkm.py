import torch
import triton
from triton import language as tl

from flash_moe.ops.utils import ensure_contiguous


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=4),
    ],
    key=["num_tokens", "hidden_size", "topk"],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["num_tokens"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["hidden_size"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _pkm_fwd_kernel(
    VALUES,
    ROUTING_WEIGHTS,
    INDICES,
    OUTPUT,
    stride_ve,
    stride_vn,
    stride_rm,
    stride_rk,
    stride_im,
    stride_ik,
    stride_om,
    stride_on,
    num_tokens,
    hidden_size,
    topk: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Initialize offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Mask for valid tokens and hidden dimensions
    mask_m = offs_m < num_tokens
    mask_n = offs_n < hidden_size

    # Initialize accumulator: (BLOCK_M, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over topk experts
    for k in range(topk):
        # Get the expert index for this k: (BLOCK_M,)
        if EVEN_M:
            expert_idx = tl.load(INDICES + offs_m * stride_im + k * stride_ik)
        else:
            expert_idx = tl.load(
                INDICES + offs_m * stride_im + k * stride_ik,
                mask=mask_m,
                other=0,
            )

        # Get the routing weight for this k: (BLOCK_M,)
        if EVEN_M:
            weight = tl.load(ROUTING_WEIGHTS + offs_m * stride_rm + k * stride_rk)
        else:
            weight = tl.load(
                ROUTING_WEIGHTS + offs_m * stride_rm + k * stride_rk,
                mask=mask_m,
                other=0.0,
            )

        # Load values for this expert: (BLOCK_M, BLOCK_N)
        # values[expert_idx, offs_n]
        values_ptrs = (
            VALUES + expert_idx[:, None] * stride_ve + offs_n[None, :] * stride_vn
        )
        if EVEN_M:
            if EVEN_N:
                values = tl.load(values_ptrs)
            else:
                values = tl.load(values_ptrs, mask=mask_n[None, :], other=0.0)
        else:
            if EVEN_N:
                values = tl.load(values_ptrs, mask=mask_m[:, None], other=0.0)
            else:
                values = tl.load(
                    values_ptrs,
                    mask=mask_m[:, None] & mask_n[None, :],
                    other=0.0,
                )

        # Accumulate weighted values
        acc += weight[:, None] * values.to(tl.float32)

    # Store output: (BLOCK_M, BLOCK_N)
    output_ptrs = OUTPUT + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    if EVEN_M:
        if EVEN_N:
            tl.store(output_ptrs, acc.to(OUTPUT.dtype.element_ty))
        else:
            tl.store(output_ptrs, acc.to(OUTPUT.dtype.element_ty), mask=mask_n[None, :])
    else:
        if EVEN_N:
            tl.store(output_ptrs, acc.to(OUTPUT.dtype.element_ty), mask=mask_m[:, None])
        else:
            tl.store(
                output_ptrs,
                acc.to(OUTPUT.dtype.element_ty),
                mask=mask_m[:, None] & mask_n[None, :],
            )


def pkm_forward(
    values: torch.Tensor,
    routing_weights: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """
    PKM forward pass using Triton kernel.

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

    # Launch kernel
    def grid(META):
        return (
            triton.cdiv(num_tokens, META["BLOCK_M"]),
            triton.cdiv(hidden_size, META["BLOCK_N"]),
        )

    _pkm_fwd_kernel[grid](
        values,
        routing_weights,
        indices,
        output,
        values.stride(0),
        values.stride(1),
        routing_weights.stride(0),
        routing_weights.stride(1),
        indices.stride(0),
        indices.stride(1),
        output.stride(0),
        output.stride(1),
        num_tokens,
        hidden_size,
        topk,
    )

    return output


class TritonPKMFunc(torch.autograd.Function):
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
        raise NotImplementedError("TritonPKMFunc backward is not implemented yet.")


def triton_pkm_func(
    values: torch.Tensor,
    routing_weights: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    return TritonPKMFunc.apply(values, routing_weights, indices)
