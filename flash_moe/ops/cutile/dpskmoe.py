# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import cuda.tile as ct

from flash_moe.ops.utils import next_power_of_2, ensure_contiguous


ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


def swizzle_2d_from_bid(M, N, tm, tn, GROUP_SIZE_M, bid):
    # Get the global IDs of a given block in a 1D grid.
    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + (bid % group_size_m)
    bid_n = (bid % num_bid_in_group) // group_size_m
    return bid_m, bid_n


def swizzle_2d(M, N, tm, tn, GROUP_SIZE_M):
    # Get the global IDs of the current block in a 1D grid.
    bid = ct.bid(0)
    return swizzle_2d_from_bid(M, N, tm, tn, GROUP_SIZE_M, bid)


@ct.kernel
def _silu_and_mul_kernel(A, B, C, TILE_N: ConstInt):
    """
    Element-wise kernel that computes SiLU(A) * B.

    Args:
        A: Input tensor A.
        B: Input tensor B.
        C: Output tensor.
    """

    bid_m = ct.bid(0)
    ta = ct.load(A, (bid_m, 0), (1, TILE_N)).astype(ct.float32)
    tb = ct.load(B, (bid_m, 0), (1, TILE_N)).astype(ct.float32)

    # Sigmoid(ta)
    denom = ct.add(1, ct.exp(-ta), flush_to_zero=True)
    sigmoid_ta = ct.truediv(
        1.0, denom, flush_to_zero=True, rounding_mode=ct.RoundingMode.APPROX
    )

    # SiLU(ta) * tb
    silu_ta = ct.mul(ta, sigmoid_ta, flush_to_zero=True)
    tc = ct.mul(silu_ta, tb, flush_to_zero=True)

    ct.store(C, (bid_m, 0), tc.astype(C.dtype))


@ct.kernel
def _fwd_kernel(
    A,
    B,
    C,
    topk_weights,
    sorted_token_ids,
    sorted_expert_ids,
    num_token_replicas: int,
    mul_routed_weight: ConstBool,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    TILE_K: ConstInt,
):
    """
    Fused MoE kernel that multiplies tokens by their assigned expert weights.

    Args:
        A: Input tokens, shape (batch, K).
        B: Expert weights, shape (num_experts, N, K).
        C: Output tensor, shape (num_tokens * topk, N).
        topk_weights: Router weights for each token-expert pair, shape (num_tokens * topk,).
        sorted_token_ids: Token indices sorted by expert assignment, replicated topk times,
            and padded to align with TILE_M.
        sorted_expert_ids: Expert index for each TILE_M, sorted.
        num_token_replicas: Replication factor applied to each token row in A (topk or 1).
        mul_routed_weight: Whether to multiply output by router weights.

    Token ids are sorted and padded to ensure each expert processes a multiple of TILE_M tokens,
    enabling efficient tiled matrix multiplication.
    """

    M = sorted_token_ids.shape[0]
    N = B.shape[1]
    K = B.shape[2]

    GROUP_SIZE_M = 8
    bid_m, bid_n = swizzle_2d(M, N, TILE_M, TILE_N, GROUP_SIZE_M)

    zero_pad = ct.PaddingMode.ZERO

    # Gather replicated/padded token indices handled by this block pair (bid_m, bid_n).
    token_id_indices = bid_m * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    token_ids = ct.gather(sorted_token_ids, token_id_indices)

    # Collapse the replica dimension to recover the source row in A for each entry.
    a_row_indices = token_ids // num_token_replicas

    # Each TILE_M block is homogenous in expert assignment; fetch the expert id once.
    expert_id = ct.load(sorted_expert_ids, index=bid_m, shape=())

    accumulator = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
    for k in range(0, ct.cdiv(K, TILE_K)):
        a_col_indices = k * TILE_K + ct.arange(TILE_K, dtype=ct.int32)
        a = ct.gather(A, (a_row_indices[:, None], a_col_indices[None, :]))

        b = ct.load(
            B,
            (expert_id, k, bid_n),
            shape=(1, TILE_K, TILE_N),
            order=(0, 2, 1),
            padding_mode=zero_pad,
        ).reshape((TILE_K, TILE_N))

        accumulator = ct.mma(a, b, accumulator)

    if mul_routed_weight:
        moe_weight = ct.gather(topk_weights, token_ids)
        accumulator = accumulator * moe_weight[:, None]

    # Compute the column span this block covers and scatter the tile back into C.
    c_col_indices = bid_n * TILE_N + ct.arange(TILE_N, dtype=ct.int32)
    accumulator = ct.astype(accumulator, C.dtype)
    ct.scatter(C, (token_ids[:, None], c_col_indices[None, :]), accumulator)


def silu_and_mul_kernel(AB: torch.Tensor, C: torch.Tensor):
    A, B = AB.chunk(2, dim=-1)
    ct.launch(
        torch.cuda.current_stream(),
        (AB.shape[0],),
        _silu_and_mul_kernel,
        (A, B, C, next_power_of_2(C.shape[-1])),
    )


def fwd_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    mul_routed_weight: bool,
    num_token_replicas: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
) -> None:
    m = sorted_token_ids.shape[0]
    n = B.shape[1]

    grid = (math.ceil(m / tile_m) * math.ceil(n / tile_n),)
    topk_weights = topk_weights.view(-1)
    C = C.view(-1, C.shape[2])

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _fwd_kernel,
        (
            A,
            B,
            C,
            topk_weights,
            sorted_token_ids,
            sorted_expert_ids,
            num_token_replicas,
            mul_routed_weight,
            tile_m,
            tile_n,
            tile_k,
        ),
    )


def moe_align_tile_size_torch(topk_ids: torch.Tensor, tile_m: int, num_experts: int):
    """
    Sort, replicate, and pad token indices by expert so every expert processes a
    TILE_M-aligned tile when launching the fused_moe_kernel.

    Args:
        topk_ids: Router-selected expert ids per token (num_tokens, topk).
        tile_m: Tile size used along the M dimension by the kernel.
        num_experts: Total number of experts present in w1/w2 tensors.

    Returns:
        sorted_token_ids: 1-D tensor containing the flattened token-replica indices
            sorted by expert; remaining slots are filled with a sentinel index
            (num_tokens * topk) for padding.
        sorted_expert_ids: For each block, the expert id that
            owns the corresponding TILE_M slice in `sorted_token_ids`.
    """

    device = topk_ids.device
    num_tokens, topk = topk_ids.shape
    total_tokens = num_tokens * topk

    # Flatten expert ids (num_tokens * topk) and sort by experts.
    flat_expert_ids = topk_ids.reshape(-1)
    sorted_token_indices = torch.argsort(flat_expert_ids, stable=True)

    # Determine how many replicas each expert owns and how many TILE_M blocks we need
    # once padded to TILE_M alignment.
    expert_token_counts = torch.bincount(flat_expert_ids, minlength=num_experts)
    expert_block_counts = (expert_token_counts - 1 + tile_m) // tile_m
    total_blocks = expert_block_counts.sum()

    # Allocate output buffers; fill token ids with sentinel value (total_tokens).
    sorted_token_ids = torch.full(
        (total_blocks * tile_m,), total_tokens, device=device, dtype=torch.int32
    )

    sorted_expert_ids = torch.zeros((total_blocks,), device=device, dtype=torch.int32)

    current_block = 0
    current_token = 0
    for expert_id in range(num_experts):
        token_count = expert_token_counts[expert_id]
        block_count = expert_block_counts[expert_id]

        # Map each TILE_M block with its owning expert id
        sorted_expert_ids[current_block : current_block + block_count] = expert_id

        sorted_token_start = current_block * tile_m
        # Copy the expert's sorted token indices; residual slots remain at the
        # sentinel value for padding.
        sorted_token_ids[sorted_token_start : sorted_token_start + token_count] = (
            sorted_token_indices[current_token : current_token + token_count]
        )

        current_token += token_count
        current_block += block_count

    return sorted_token_ids, sorted_expert_ids


def dpskmoe_forward(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Executes a Mixture-of-Experts (MoE) forward pass using the fused cuTile kernel.

    Args:
        hidden_states: Token activations, shape (num_tokens, hidden_size)
        w1: Expert gate+up projection weights,
            shape (num_experts, intermediate_size * 2, hidden_size)
        w2: Expert down projection weights,
            shape (num_experts, hidden_size, intermediate_size)
        topk_weights: Router weights per token, shape (num_tokens, topk)
        topk_ids: Expert indices per token, shape (num_tokens, topk)

    Returns:
        Tensor with the same shape/dtype as `hidden_states`.
    """
    out_dtype = hidden_states.dtype
    device = hidden_states.device

    num_tokens, hidden_size = hidden_states.shape
    num_experts, _, intermediate_size = w2.shape
    _, topk = topk_ids.shape
    tile_m, tile_n, tile_k = 128, 128, 64

    if w1.shape[1] != intermediate_size * 2:
        raise ValueError(
            "w1 must have 2 * intermediate_size rows (gate + up projection)"
        )

    intermediate_cache1 = torch.zeros(
        (num_tokens, topk, intermediate_size * 2),
        device=device,
        dtype=out_dtype,
    )
    intermediate_cache2 = torch.zeros(
        (num_tokens * topk, intermediate_size),
        device=device,
        dtype=out_dtype,
    )
    intermediate_cache3 = torch.zeros(
        (num_tokens, topk, hidden_size),
        device=device,
        dtype=out_dtype,
    )

    sorted_token_ids, sorted_expert_ids = moe_align_tile_size_torch(
        topk_ids,
        tile_m,
        num_experts,
    )

    fwd_kernel(
        hidden_states,
        w1,
        intermediate_cache1,
        topk_weights,
        sorted_token_ids,
        sorted_expert_ids,
        mul_routed_weight=False,
        num_token_replicas=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
    )

    silu_and_mul_kernel(
        intermediate_cache1.view(-1, intermediate_cache1.shape[-1]),
        intermediate_cache2,
    )

    fwd_kernel(
        intermediate_cache2,
        w2,
        intermediate_cache3,
        topk_weights,
        sorted_token_ids,
        sorted_expert_ids,
        mul_routed_weight=True,
        num_token_replicas=1,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
    )

    output = torch.sum(intermediate_cache3, dim=1)
    return output


class CutileDPSKMoEFunc(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        output = dpskmoe_forward(
            hidden_states,
            torch.cat([gate_proj, up_proj], dim=1),
            down_proj,
            topk_weights,
            topk_ids,
        )

        ctx.save_for_backward(
            hidden_states,
            gate_proj,
            up_proj,
            down_proj,
            topk_weights,
            topk_ids,
        )

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        raise NotImplementedError("CuTileDPSKMoEFunc backward is not implemented yet.")


def cutile_dpskmoe_func(
    hidden_states: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    return CutileDPSKMoEFunc.apply(
        hidden_states,
        gate_proj,
        up_proj,
        down_proj,
        topk_weights,
        topk_ids,
    )
