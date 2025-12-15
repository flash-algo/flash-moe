import torch
import triton
from triton import language as tl

from flash_moe.ops.utils import ensure_contiguous


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=4),
    ],
    key=["num_experts", "num_experts_per_tok"],
)
@triton.jit
def _fwd_kernel(
    ROUTER_LOGITS,
    SCORES,
    INDICES,
    stride_rs,
    stride_rt,
    stride_rk,
    stride_st,
    stride_sk,
    stride_it,
    stride_ik,
    num_expert_keys: tl.constexpr,
    num_experts_per_tok: tl.constexpr,
    num_experts: tl.constexpr,
):
    start_token = tl.program_id(0)

    # Initialize offsets
    offs_e = tl.arange(0, num_experts)
    mask = offs_e < num_experts

    # Compute offsets
    ix = offs_e // num_expert_keys
    iy = offs_e - ix * num_expert_keys

    # Initialize pointers
    router_logits_x_ptrs = (
        ROUTER_LOGITS + 0 * stride_rs + start_token * stride_rt + ix * stride_rk
    )
    router_logits_y_ptrs = (
        ROUTER_LOGITS + 1 * stride_rs + start_token * stride_rt + iy * stride_rk
    )
    scores_ptr = SCORES + start_token * stride_st
    indices_ptr = INDICES + start_token * stride_it

    # Load scores
    scores_x = tl.load(router_logits_x_ptrs, mask=mask, other=-float("inf"))
    scores_y = tl.load(router_logits_y_ptrs, mask=mask, other=-float("inf"))
    scores = scores_x + scores_y

    # Loop to find top-k experts
    for k in range(num_experts_per_tok):
        topk_scores = tl.max(scores, axis=0)
        topk_indices = tl.argmax(scores, axis=0)

        tl.store(scores_ptr + k * stride_sk, topk_scores)
        tl.store(indices_ptr + k * stride_ik, topk_indices)

        scores = tl.where(offs_e == topk_indices, -float("inf"), scores)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_EXPERTS": 4096}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_EXPERTS": 8192}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_EXPERTS": 16384}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_EXPERTS": 32768}, num_warps=8, num_stages=4),
    ],
    key=["num_experts", "num_experts_per_tok"],
)
@triton.heuristics(
    {
        "EVEN_EXPERTS": lambda args: args["num_experts"] % args["BLOCK_EXPERTS"] == 0,
    }
)
@triton.jit
def _fwd_kernel_split_experts(
    ROUTER_LOGITS,
    SCORES,
    INDICES,
    stride_rs,
    stride_rt,
    stride_rk,
    stride_st,
    stride_sk,
    stride_it,
    stride_ik,
    num_expert_keys: tl.constexpr,
    num_experts_per_tok: tl.constexpr,
    num_experts: tl.constexpr,
    EVEN_EXPERTS: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
):
    start_token = tl.program_id(0)

    # Initialize offsets
    offs_eb = tl.arange(0, BLOCK_EXPERTS)
    offs_k = tl.arange(0, num_experts_per_tok)

    # Initialize pointers
    router_logits_x_ptr = ROUTER_LOGITS + 0 * stride_rs + start_token * stride_rt
    router_logits_y_ptr = ROUTER_LOGITS + 1 * stride_rs + start_token * stride_rt
    scores_ptr = SCORES + start_token * stride_st + offs_k * stride_sk
    indices_ptr = INDICES + start_token * stride_it + offs_k * stride_ik

    # Initialize scores and indices
    scores = tl.full((num_experts_per_tok,), -float("inf"), dtype=tl.float32)
    indices = tl.full((num_experts_per_tok,), -1, dtype=tl.int64)

    # Loop over experts
    for start_expert in range(0, num_experts, BLOCK_EXPERTS):
        start_expert = tl.multiple_of(start_expert, BLOCK_EXPERTS)
        offs_e = offs_eb + start_expert

        # Compute offsets
        ix = offs_e // num_expert_keys
        iy = offs_e - ix * num_expert_keys

        # Load scores
        if EVEN_EXPERTS:
            score_x = tl.load(router_logits_x_ptr + ix * stride_rk)
            score_y = tl.load(router_logits_y_ptr + iy * stride_rk)
        else:
            score_x = tl.load(
                router_logits_x_ptr + ix * stride_rk,
                mask=offs_e < num_experts,
                other=-float("inf"),
            )
            score_y = tl.load(
                router_logits_y_ptr + iy * stride_rk,
                mask=offs_e < num_experts,
                other=-float("inf"),
            )
        score = score_x + score_y

        # Initialize scores and indices for this block
        topk_scores = tl.full((num_experts_per_tok,), -float("inf"), dtype=score.dtype)
        topk_indices = tl.full((num_experts_per_tok,), -1, dtype=tl.int64)

        # Loop to find top-k experts in this block
        for k in tl.static_range(num_experts_per_tok):
            max_score = tl.max(score, axis=0)
            max_index = tl.argmax(score, axis=0)
            topk_scores = tl.where(offs_k == k, max_score, topk_scores)
            topk_indices = tl.where(offs_k == k, max_index + start_expert, topk_indices)
            score = tl.where(offs_eb == max_index, -float("inf"), score)

        # Initialize new scores and indices after merging
        new_topk_scores = tl.full(
            (num_experts_per_tok,), -float("inf"), dtype=score.dtype
        )
        new_topk_indices = tl.full((num_experts_per_tok,), -1, dtype=tl.int64)

        # Merge with previous top-k
        scores = scores.to(score.dtype)
        for k in tl.static_range(num_experts_per_tok):
            max_score = tl.max(scores, axis=0)
            max_index = tl.argmax(scores, axis=0)
            max_topk_score = tl.max(topk_scores, axis=0)
            max_topk_index = tl.argmax(topk_scores, axis=0)

            take_from_scores = max_score >= max_topk_score
            cand_scores_idx = tl.where(offs_k == max_index, indices, -1)
            max_scores_idx = tl.max(cand_scores_idx, axis=0)

            cand_topk_idx = tl.where(offs_k == max_topk_index, topk_indices, -1)
            max_topk_idx = tl.max(cand_topk_idx, axis=0)

            chosen_score = tl.where(take_from_scores, max_score, max_topk_score)
            chosen_index = tl.where(take_from_scores, max_scores_idx, max_topk_idx)

            new_topk_scores = tl.where(offs_k == k, chosen_score, new_topk_scores)
            new_topk_indices = tl.where(offs_k == k, chosen_index, new_topk_indices)

            scores = tl.where(
                (offs_k == max_index) & (take_from_scores),
                -float("inf"),
                scores,
            )
            topk_scores = tl.where(
                (offs_k == max_topk_index) & (~take_from_scores),
                -float("inf"),
                topk_scores,
            )
        scores = new_topk_scores
        indices = new_topk_indices

    # Write back scores and indices
    tl.store(scores_ptr, scores)
    tl.store(indices_ptr, indices)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=["num_expert_keys", "num_experts_per_tok"],
)
@triton.jit
def _bwd_kernel(
    DSCORES,
    INDICES,
    DROUTER_LOGITS,
    stride_dst,
    stride_dsk,
    stride_it,
    stride_ik,
    stride_drs,
    stride_drt,
    stride_drk,
    num_expert_keys: tl.constexpr,
    num_experts_per_tok: tl.constexpr,
):
    start_token = tl.program_id(0)

    # Initialize offsets
    offs_k = tl.arange(0, num_experts_per_tok)
    mask = offs_k < num_experts_per_tok

    # Initialize pointers
    dscores_ptr = DSCORES + start_token * stride_dst + offs_k * stride_dsk
    indices_ptr = INDICES + start_token * stride_it + offs_k * stride_ik
    drouter_logits_x_ptr = DROUTER_LOGITS + 0 * stride_drs + start_token * stride_drt
    drouter_logits_y_ptr = DROUTER_LOGITS + 1 * stride_drs + start_token * stride_drt

    # Load dscores and indices
    dscores = tl.load(dscores_ptr, mask=mask, other=0.0)
    indices = tl.load(indices_ptr, mask=mask, other=0)

    # Compute offsets
    ix = indices // num_expert_keys
    iy = indices - ix * num_expert_keys

    # Atomic add to drouter_logits
    tl.atomic_add(
        drouter_logits_x_ptr + ix * stride_drk,
        dscores,
        mask=mask,
    )
    tl.atomic_add(
        drouter_logits_y_ptr + iy * stride_drk,
        dscores,
        mask=mask,
    )


def _flash_router_forward(
    router_logits: torch.Tensor,
    num_expert_keys: int,
    num_experts_per_tok: int,
):
    assert router_logits.dim() == 3, (
        "router_logits must be a 3D tensor of shape (2, batch_size * seq_len, num_expert_keys)"
    )
    assert router_logits.size(0) == 2, "The first dimension of router_logits must be 2"
    assert 0 <= num_experts_per_tok <= num_expert_keys, (
        f"num_experts_per_tok should be in [0, {num_expert_keys}], but got {num_experts_per_tok}"
    )
    num_tokens = router_logits.size(1)

    scores = torch.empty(
        (num_tokens, num_experts_per_tok),
        device=router_logits.device,
        dtype=router_logits.dtype,
    )
    indices = torch.empty(
        (num_tokens, num_experts_per_tok),
        device=router_logits.device,
        dtype=torch.int64,
    )

    num_experts = triton.next_power_of_2(num_expert_keys * num_expert_keys)

    grid = (num_tokens,)

    if num_expert_keys <= 128:
        _fwd_kernel[grid](
            router_logits,
            scores,
            indices,
            router_logits.stride(0),
            router_logits.stride(1),
            router_logits.stride(2),
            scores.stride(0),
            scores.stride(1),
            indices.stride(0),
            indices.stride(1),
            num_expert_keys,
            num_experts_per_tok,
            num_experts,
        )
    else:
        _fwd_kernel_split_experts[grid](
            router_logits,
            scores,
            indices,
            router_logits.stride(0),
            router_logits.stride(1),
            router_logits.stride(2),
            scores.stride(0),
            scores.stride(1),
            indices.stride(0),
            indices.stride(1),
            num_expert_keys,
            num_experts_per_tok,
            num_experts,
        )

    return scores, indices


def _flash_router_backward(
    dscores: torch.Tensor,
    indices: torch.Tensor,
    num_expert_keys: int,
):
    num_tokens, num_experts_per_tok = dscores.shape

    drouter_logits = torch.zeros(
        (2, num_tokens, num_expert_keys), device=dscores.device, dtype=dscores.dtype
    )
    grid = (num_tokens,)

    _bwd_kernel[grid](
        dscores,
        indices,
        drouter_logits,
        dscores.stride(0),
        dscores.stride(1),
        indices.stride(0),
        indices.stride(1),
        drouter_logits.stride(0),
        drouter_logits.stride(1),
        drouter_logits.stride(2),
        num_expert_keys,
        num_experts_per_tok,
    )

    return drouter_logits


class FlashRouterFunc(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, router_logits, num_expert_keys, num_experts_per_tok):
        """
        Args:
            router_logits: (2, batch_size * seq_len, num_expert_keys)
            num_expert_keys: int
            num_experts_per_tok: int

        Returns:
            scores: (batch_size * seq_len, num_experts_per_tok)
            indices: (batch_size * seq_len, num_experts_per_tok)
        """

        scores, indices = _flash_router_forward(
            router_logits, num_expert_keys, num_experts_per_tok
        )
        ctx.save_for_backward(indices)
        ctx.num_expert_keys = num_expert_keys

        return scores, indices

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dscores, dindices):
        (indices,) = ctx.saved_tensors

        drouter_logits = _flash_router_backward(dscores, indices, ctx.num_expert_keys)

        # No gradients for indices
        return drouter_logits, None, None


def triton_flash_router_func(router_logits, num_expert_keys, num_experts_per_tok):
    return FlashRouterFunc.apply(router_logits, num_expert_keys, num_experts_per_tok)
