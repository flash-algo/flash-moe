import torch
import triton
import triton.language as tl

from flash_moe.ops.utils import next_power_of_2, ensure_contiguous


@triton.jit
def _silu_and_mul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    N: tl.constexpr,
    stride_ab: tl.constexpr,
    stride_c: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Element-wise kernel that computes SiLU(A) * B.
    Each program handles one row.
    """
    row_idx = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_N)
    mask = col_offsets < N

    a_ptrs = A_ptr + row_idx * stride_ab + col_offsets
    b_ptrs = B_ptr + row_idx * stride_ab + col_offsets

    a = tl.load(a_ptrs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptrs, mask=mask, other=0.0).to(tl.float32)

    # SiLU(a) = a * sigmoid(a)
    sigmoid_a = tl.sigmoid(a)
    silu_a = a * sigmoid_a
    c = silu_a * b

    c_ptrs = C_ptr + row_idx * stride_c + col_offsets
    tl.store(c_ptrs, c.to(C_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _fwd_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    sorted_expert_ids_ptr,
    stride_a_m,
    stride_a_k,
    stride_b_e,
    stride_b_n,
    stride_b_k,
    stride_c_m,
    stride_c_n,
    num_token_replicas: int,
    num_valid_tokens: int,
    M: int,
    N: int,
    K: int,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Fused MoE kernel that multiplies tokens by their assigned expert weights.

    Args:
        A: Input tokens, shape (num_tokens, K).
        B: Expert weights, shape (num_experts, N, K).
        C: Output tensor, shape (num_tokens * topk, N).
        topk_weights: Router weights for each token-expert pair.
        sorted_token_ids: Token indices sorted by expert assignment.
        sorted_expert_ids: Expert index for each TILE_M block.
        num_token_replicas: Replication factor (topk or 1).
    """
    pid = tl.program_id(0)

    num_tiles_m = tl.cdiv(M, TILE_M)
    num_tiles_n = tl.cdiv(N, TILE_N)

    # Swizzle for better L2 cache utilization
    num_tiles_in_group = GROUP_SIZE_M * num_tiles_n
    group_id = pid // num_tiles_in_group
    first_tile_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_tiles_m - first_tile_m, GROUP_SIZE_M)
    tile_m = first_tile_m + (pid % group_size_m)
    tile_n = (pid % num_tiles_in_group) // group_size_m

    # Load expert id for this tile
    expert_id = tl.load(sorted_expert_ids_ptr + tile_m)

    # Compute token indices for this tile
    offs_m = tile_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = tile_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    # Load sorted token ids
    token_ids = tl.load(sorted_token_ids_ptr + offs_m)

    # Collapse replica dimension to get source row in A
    a_row_indices = token_ids // num_token_replicas

    # Initialize accumulator
    accumulator = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    # Main loop over K dimension
    for k in range(0, tl.cdiv(K, TILE_K)):
        k_start = k * TILE_K
        k_offs = k_start + offs_k

        # Load A tile using gather (indirect indexing)
        # A shape: (num_tokens, K)
        a_ptrs = (
            A_ptr + a_row_indices[:, None] * stride_a_m + k_offs[None, :] * stride_a_k
        )
        a_mask = (a_row_indices[:, None] >= 0) & (k_offs[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile
        # B shape: (num_experts, N, K)
        # We need B[expert_id, offs_n, k_offs]
        b_ptrs = (
            B_ptr
            + expert_id * stride_b_e
            + offs_n[:, None] * stride_b_n
            + k_offs[None, :] * stride_b_k
        )
        b_mask = (offs_n[:, None] < N) & (k_offs[None, :] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Matrix multiply: (TILE_M, TILE_K) @ (TILE_K, TILE_N)
        accumulator += tl.dot(a, tl.trans(b))

    # Apply routing weights if needed
    if MUL_ROUTED_WEIGHT:
        moe_weights = tl.load(topk_weights_ptr + token_ids)
        accumulator = accumulator * moe_weights[:, None]

    # Store result using scatter
    c_ptrs = C_ptr + token_ids[:, None] * stride_c_m + offs_n[None, :] * stride_c_n
    c_mask = (token_ids[:, None] < num_valid_tokens) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator.to(C_ptr.dtype.element_ty), mask=c_mask)


def silu_and_mul_kernel(AB: torch.Tensor, C: torch.Tensor):
    """Launch the SiLU and multiply kernel."""
    A, B = AB.chunk(2, dim=-1)
    num_rows = A.shape[0]
    N = A.shape[1]

    # A and B from chunk share the same row stride (which is AB's row stride)
    stride_ab = AB.stride(0)
    stride_c = C.stride(0)

    # Choose BLOCK_N as next power of 2
    BLOCK_N = next_power_of_2(N)

    grid = (num_rows,)
    _silu_and_mul_kernel[grid](
        A,
        B,
        C,
        N,
        stride_ab,
        stride_c,
        BLOCK_N,
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
    tile_m: int = 64,
    tile_n: int = 64,
    tile_k: int = 32,
) -> None:
    """Launch the fused MoE forward kernel."""
    M = sorted_token_ids.shape[0]
    N = B.shape[1]
    K = B.shape[2]

    topk_weights_flat = topk_weights.view(-1)
    C_flat = C.view(-1, C.shape[-1])
    num_valid_tokens = C_flat.shape[0]

    grid = (triton.cdiv(M, tile_m) * triton.cdiv(N, tile_n),)

    _fwd_kernel[grid](
        A,
        B,
        C_flat,
        topk_weights_flat,
        sorted_token_ids,
        sorted_expert_ids,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        C_flat.stride(0),
        C_flat.stride(1),
        num_token_replicas,
        num_valid_tokens,
        M,
        N,
        K,
        mul_routed_weight,
        tile_m,
        tile_n,
        tile_k,
        GROUP_SIZE_M=8,
    )


def moe_align_tile_size_torch(topk_ids: torch.Tensor, tile_m: int, num_experts: int):
    """
    Sort, replicate, and pad token indices by expert so every expert processes a
    TILE_M-aligned tile.

    Args:
        topk_ids: Router-selected expert ids per token (num_tokens, topk).
        tile_m: Tile size used along the M dimension by the kernel.
        num_experts: Total number of experts present in w1/w2 tensors.

    Returns:
        sorted_token_ids: 1-D tensor containing the flattened token-replica indices
            sorted by expert.
        sorted_expert_ids: For each block, the expert id that owns the corresponding
            TILE_M slice.
    """
    device = topk_ids.device
    num_tokens, topk = topk_ids.shape
    total_tokens = num_tokens * topk

    # Flatten expert ids and sort by experts
    flat_expert_ids = topk_ids.reshape(-1)
    sorted_token_indices = torch.argsort(flat_expert_ids, stable=True)

    # Count tokens per expert and compute block counts
    expert_token_counts = torch.bincount(flat_expert_ids, minlength=num_experts)
    expert_block_counts = (expert_token_counts + tile_m - 1) // tile_m
    total_blocks = expert_block_counts.sum().item()

    # Allocate output buffers
    sorted_token_ids = torch.full(
        (total_blocks * tile_m,), total_tokens, device=device, dtype=torch.int32
    )
    sorted_expert_ids = torch.zeros((total_blocks,), device=device, dtype=torch.int32)

    current_block = 0
    current_token = 0
    for expert_id in range(num_experts):
        token_count = expert_token_counts[expert_id].item()
        block_count = expert_block_counts[expert_id].item()

        # Map each TILE_M block with its owning expert id
        sorted_expert_ids[current_block : current_block + block_count] = expert_id

        sorted_token_start = current_block * tile_m
        # Copy the expert's sorted token indices
        sorted_token_ids[sorted_token_start : sorted_token_start + token_count] = (
            sorted_token_indices[current_token : current_token + token_count].to(
                torch.int32
            )
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
    Executes a Mixture-of-Experts (MoE) forward pass using fused Triton kernels.

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
    tile_m, tile_n, tile_k = 64, 64, 32

    if w1.shape[1] != intermediate_size * 2:
        raise ValueError(
            "w1 must have 2 * intermediate_size rows (gate + up projection)"
        )

    # Allocate intermediate buffers
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

    # Sort and align tokens by expert
    sorted_token_ids, sorted_expert_ids = moe_align_tile_size_torch(
        topk_ids,
        tile_m,
        num_experts,
    )

    # First matmul: hidden_states @ w1^T -> intermediate_cache1
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

    # SiLU activation and multiply
    silu_and_mul_kernel(
        intermediate_cache1.view(-1, intermediate_cache1.shape[-1]),
        intermediate_cache2,
    )

    # Second matmul: intermediate_cache2 @ w2^T -> intermediate_cache3
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

    # Sum over topk dimension
    output = torch.sum(intermediate_cache3, dim=1)
    return output


class TritonDPSKMoEFunc(torch.autograd.Function):
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
        raise NotImplementedError("TritonDPSKMoEFunc backward is not implemented yet.")


def triton_dpskmoe_func(
    hidden_states: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    return TritonDPSKMoEFunc.apply(
        hidden_states,
        gate_proj,
        up_proj,
        down_proj,
        topk_weights,
        topk_ids,
    )
