from typing import Optional
import torch
import triton
from triton import language as tl


def _next_power_of_two(x: int) -> int:
    return 1 << (x - 1).bit_length()


def maybe_contiguous(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def round_multiple(x, m):
    return (x + m - 1) // m * m


@triton.jit
def _fwd_kernel(
    router_logits_ptr,
    scores_ptr,
    indices_ptr,
    num_keys: tl.constexpr,
    top_k: tl.constexpr,
    stride_rs,
    stride_rt,
    stride_rk,
    stride_st,
    stride_sk,
    stride_it,
    stride_ik,
    BLOCK_SIZE: tl.constexpr,
):
    start_token = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    k2 = num_keys * num_keys
    mask = offs < k2

    ix = offs // num_keys
    iy = offs - ix * num_keys

    x_ptrs = (
        router_logits_ptr + 0 * stride_rs + start_token * stride_rt + ix * stride_rk
    )
    y_ptrs = (
        router_logits_ptr + 1 * stride_rs + start_token * stride_rt + iy * stride_rk
    )

    scores = tl.load(x_ptrs, mask=mask, other=-float("inf")) + tl.load(
        y_ptrs, mask=mask, other=-float("inf")
    )
    scores = tl.where(mask, scores, -float("inf"))

    for k in range(top_k):
        topk_scores = tl.max(scores, axis=0)
        topk_indices = tl.argmax(scores, axis=0)

        tl.store(
            scores_ptr + start_token * stride_st + k * stride_sk,
            topk_scores,
        )
        tl.store(
            indices_ptr + start_token * stride_it + k * stride_ik,
            topk_indices,
        )
        scores = tl.where(offs == topk_indices, -float("inf"), scores)


@triton.jit
def _bwd_kernel(
    dscores_ptr,
    indices_ptr,
    drouter_logits_ptr,
    num_keys: tl.constexpr,
    top_k: tl.constexpr,
    stride_dst,
    stride_dsk,
    stride_it,
    stride_ik,
    stride_drs,
    stride_drt,
    stride_drk,
):
    start_token = tl.program_id(0)
    offs = tl.arange(0, top_k)
    mask = offs < top_k

    dscores = tl.load(
        dscores_ptr + start_token * stride_dst + offs * stride_dsk,
        mask=mask,
        other=0.0,
    )
    indices = tl.load(
        indices_ptr + start_token * stride_it + offs * stride_ik,
        mask=mask,
        other=0,
    )

    ix = indices // num_keys
    iy = indices - ix * num_keys

    tl.atomic_add(
        drouter_logits_ptr
        + 0 * stride_drs
        + start_token * stride_drt
        + ix * stride_drk,
        dscores,
        mask=mask,
    )
    tl.atomic_add(
        drouter_logits_ptr
        + 1 * stride_drs
        + start_token * stride_drt
        + iy * stride_drk,
        dscores,
        mask=mask,
    )


def _flash_router_forward(
    router_logits: torch.Tensor,
    num_keys: int,
    top_k: int,
):
    assert router_logits.dim() == 3, (
        "router_logits must be a 3D tensor of shape (2, batch_size * seq_len, num_keys)"
    )
    assert router_logits.size(0) == 2, "The first dimension of router_logits must be 2"
    assert 0 <= top_k <= num_keys, (
        f"top_k should be in [0, {num_keys}], but got {top_k}"
    )
    num_tokens = router_logits.size(1)
    dtype = router_logits.dtype

    router_logits = router_logits.to(torch.float32)
    scores = torch.empty(
        (num_tokens, top_k), device=router_logits.device, dtype=router_logits.dtype
    )
    indices = torch.empty(
        (num_tokens, top_k), device=router_logits.device, dtype=torch.int64
    )

    BLOCK_SIZE = _next_power_of_two(num_keys * num_keys)
    grid = (num_tokens,)

    _fwd_kernel[grid](
        router_logits,
        scores,
        indices,
        num_keys,
        top_k,
        router_logits.stride(0),
        router_logits.stride(1),
        router_logits.stride(2),
        scores.stride(0),
        scores.stride(1),
        indices.stride(0),
        indices.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return scores.to(dtype), indices


def _flash_router_backward(
    dscores: torch.Tensor,
    indices: torch.Tensor,
    num_keys: int,
):
    num_tokens, top_k = dscores.shape

    drouter_logits = torch.zeros(
        (2, num_tokens, num_keys), device=dscores.device, dtype=dscores.dtype
    )
    grid = (num_tokens,)

    _bwd_kernel[grid](
        dscores,
        indices,
        drouter_logits,
        num_keys,
        top_k,
        dscores.stride(0),
        dscores.stride(1),
        indices.stride(0),
        indices.stride(1),
        drouter_logits.stride(0),
        drouter_logits.stride(1),
        drouter_logits.stride(2),
    )

    return drouter_logits


class FlashRouterFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, router_logits, num_keys, top_k):
        """
        Args:
            router_logits: (2, batch_size * seq_len, num_keys)
            num_keys: int
            top_k: int

        Returns:
            scores: (batch_size * seq_len, top_k)
            indices: (batch_size * seq_len, top_k)
        """
        # Make sure that the last dimension is contiguous
        router_logits = maybe_contiguous(router_logits)

        scores, indices = _flash_router_forward(router_logits, num_keys, top_k)
        ctx.save_for_backward(indices)
        ctx.num_keys = num_keys

        return scores, indices

    def backward(ctx, dscores, dindices):
        (indices,) = ctx.saved_tensors

        # Make sure that the last dimension is contiguous
        dscores = maybe_contiguous(dscores)

        drouter_logits = _flash_router_backward(dscores, indices, ctx.num_keys)

        # No gradients for indices
        return drouter_logits, None, None


def triton_flash_router_func(router_logits, num_keys, top_k):
    return FlashRouterFunc.apply(router_logits, num_keys, top_k)
