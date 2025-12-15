import torch
import triton
from triton import language as tl

from flash_moe.ops.utils import ensure_contiguous


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=8, num_stages=4),
    ],
    key=["num_tokens", "hidden_size", "num_experts_per_tok"],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["num_tokens"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["hidden_size"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _fwd_scores_kernel(
    HIDDEN_STATES,
    DOWN_WEIGHTS,
    INDICES,
    EXPERT_SCORES,
    stride_hm,
    stride_hn,
    stride_de,
    stride_dn,
    stride_im,
    stride_ik,
    stride_am,
    stride_ak,
    num_tokens,
    hidden_size: tl.constexpr,
    num_experts_per_tok: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    # Initialize offsets
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, num_experts_per_tok)

    # Initialize pointers
    hidden_states_ptr = HIDDEN_STATES + offs_m[:, None] * stride_hm
    indices_ptr = INDICES + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
    expert_scores_ptr = (
        EXPERT_SCORES + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    )

    # Load indices
    if EVEN_M:
        indices = tl.load(indices_ptr)
    else:
        indices = tl.load(
            indices_ptr,
            mask=offs_m[:, None] < num_tokens,
            other=0,
        )

    # Initialize weights pointers
    down_weights_ptr = DOWN_WEIGHTS + indices[:, :, None] * stride_de

    # Initialize expert scores accumulator
    acc_s = tl.zeros((BLOCK_M, num_experts_per_tok), dtype=tl.float32)

    # Loop over hidden dimension
    for start_n in range(0, hidden_size, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_nb = offs_n + start_n

        # Load hidden states
        if EVEN_M:
            if EVEN_N:
                hidden_states = tl.load(
                    hidden_states_ptr + offs_nb[None, :] * stride_hn
                )
            else:
                hidden_states = tl.load(
                    hidden_states_ptr + offs_nb[None, :] * stride_hn,
                    mask=offs_nb[None, :] < hidden_size,
                    other=0.0,
                )
        else:
            if EVEN_N:
                hidden_states = tl.load(
                    hidden_states_ptr + offs_nb[None, :] * stride_hn,
                    mask=offs_m[:, None] < num_tokens,
                    other=0.0,
                )
            else:
                hidden_states = tl.load(
                    hidden_states_ptr + offs_nb[None, :] * stride_hn,
                    mask=(offs_m[:, None] < num_tokens)
                    & (offs_nb[None, :] < hidden_size),
                    other=0.0,
                )

        # Load down weights
        if EVEN_M:
            if EVEN_N:
                down_weights = tl.load(
                    down_weights_ptr + offs_nb[None, None, :] * stride_dn
                )
            else:
                down_weights = tl.load(
                    down_weights_ptr + offs_nb[None, None, :] * stride_dn,
                    mask=offs_nb[None, None, :] < hidden_size,
                    other=0.0,
                )
        else:
            if EVEN_N:
                down_weights = tl.load(
                    down_weights_ptr + offs_nb[None, None, :] * stride_dn,
                    mask=offs_m[:, None, None] < num_tokens,
                    other=0.0,
                )
            else:
                down_weights = tl.load(
                    down_weights_ptr + offs_nb[None, None, :] * stride_dn,
                    mask=(offs_m[:, None, None] < num_tokens)
                    & (offs_nb[None, None, :] < hidden_size),
                    other=0.0,
                )

        # Compute expert scores
        acc_s += tl.sum(hidden_states[:, None, :] * down_weights, axis=2)

    # Write back expert scores
    if EVEN_M:
        tl.store(expert_scores_ptr, acc_s)
    else:
        tl.store(
            expert_scores_ptr,
            acc_s,
            mask=offs_m[:, None] < num_tokens,
        )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=4),
    ],
    key=["num_tokens", "hidden_size", "num_experts_per_tok"],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["num_tokens"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["hidden_size"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _fwd_states_kernel(
    EXPERT_SCORES,
    UP_WEIGHTS,
    INDICES,
    ROUTING_WEIGHTS,
    EXPERT_STATES,
    stride_sm,
    stride_sk,
    stride_ue,
    stride_un,
    stride_im,
    stride_ik,
    stride_rm,
    stride_rk,
    stride_om,
    stride_on,
    num_tokens,
    hidden_size: tl.constexpr,
    num_experts_per_tok: tl.constexpr,
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
    offs_k = tl.arange(0, num_experts_per_tok)

    # Initialize pointers
    expert_scores_ptr = (
        EXPERT_SCORES + offs_m[:, None] * stride_sm + offs_k[None, :] * stride_sk
    )
    indices_ptr = INDICES + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
    routing_weights_ptr = (
        ROUTING_WEIGHTS + offs_m[:, None] * stride_rm + offs_k[None, :] * stride_rk
    )
    expert_states_ptr = (
        EXPERT_STATES + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    )

    # Load indices
    if EVEN_M:
        indices = tl.load(indices_ptr)
    else:
        indices = tl.load(
            indices_ptr,
            mask=offs_m[:, None] < num_tokens,
            other=0,
        )

    # Initialize weights pointers
    up_weights_ptr = (
        UP_WEIGHTS + indices[:, :, None] * stride_ue + offs_n[None, None, :] * stride_un
    )

    # Load expert scores
    if EVEN_M:
        expert_scores = tl.load(expert_scores_ptr)
    else:
        expert_scores = tl.load(
            expert_scores_ptr,
            mask=offs_m[:, None] < num_tokens,
            other=0.0,
        )

    # Load routing weights
    if EVEN_M:
        routing_weights = tl.load(routing_weights_ptr)
    else:
        routing_weights = tl.load(
            routing_weights_ptr,
            mask=offs_m[:, None] < num_tokens,
            other=0.0,
        )

    # Compute gated weights
    gated_weights = silu(expert_scores).cast(routing_weights.dtype) * routing_weights

    # Load up weights
    if EVEN_M:
        if EVEN_N:
            up_weights = tl.load(up_weights_ptr)
        else:
            up_weights = tl.load(
                up_weights_ptr,
                mask=offs_n[None, None, :] < hidden_size,
                other=0.0,
            )
    else:
        if EVEN_N:
            up_weights = tl.load(
                up_weights_ptr,
                mask=offs_m[:, None, None] < num_tokens,
                other=0.0,
            )
        else:
            up_weights = tl.load(
                up_weights_ptr,
                mask=(offs_m[:, None, None] < num_tokens)
                & (offs_n[None, None, :] < hidden_size),
                other=0.0,
            )

    # Compute expert states
    expert_states = tl.sum(gated_weights[:, :, None] * up_weights, axis=1)

    # Write back expert states
    if EVEN_M:
        if EVEN_N:
            tl.store(expert_states_ptr, expert_states)
        else:
            tl.store(
                expert_states_ptr,
                expert_states,
                mask=offs_n[None, :] < hidden_size,
            )
    else:
        if EVEN_N:
            tl.store(
                expert_states_ptr,
                expert_states,
                mask=offs_m[:, None] < num_tokens,
            )
        else:
            tl.store(
                expert_states_ptr,
                expert_states,
                mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < hidden_size),
            )


# This version is about 20% slower but saves about 10% memory, kept as backup
# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=4),
#         triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
#         triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=4),
#     ],
#     key=["num_tokens", "hidden_size", "num_experts_per_tok"],
#     reset_to_zero=["DDOWN_WEIGHTS", "DUP_WEIGHTS"],
# )
# @triton.heuristics(
#     {
#         "EVEN_M": lambda args: args["num_tokens"] % args["BLOCK_M"] == 0,
#         "EVEN_N": lambda args: args["hidden_size"] % args["BLOCK_N"] == 0,
#     }
# )
# @triton.jit
# def _bwd_kernel(
#     DEXPERT_STATES,
#     HIDDEN_STATES,
#     DOWN_WEIGHTS,
#     UP_WEIGHTS,
#     INDICES,
#     ROUTING_WEIGHTS,
#     EXPERT_SCORES,
#     DHIDDEN_STATES,
#     DDOWN_WEIGHTS,
#     DUP_WEIGHTS,
#     DROUTING_WEIGHTS,
#     stride_dom,
#     stride_don,
#     stride_hm,
#     stride_hn,
#     stride_de,
#     stride_dn,
#     stride_ue,
#     stride_un,
#     stride_im,
#     stride_ik,
#     stride_rm,
#     stride_rk,
#     stride_sm,
#     stride_sk,
#     stride_dhm,
#     stride_dhn,
#     stride_dde,
#     stride_ddn,
#     stride_due,
#     stride_dun,
#     stride_drm,
#     stride_drk,
#     num_tokens,
#     hidden_size: tl.constexpr,
#     num_experts_per_tok: tl.constexpr,
#     EVEN_M: tl.constexpr,
#     EVEN_N: tl.constexpr,
#     BLOCK_M: tl.constexpr,
#     BLOCK_N: tl.constexpr,
# ):
#     pid_m = tl.program_id(0)

#     # Initialize offsets
#     offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
#     offs_nb = tl.arange(0, BLOCK_N)
#     offs_k = tl.arange(0, num_experts_per_tok)

#     # Initialize pointers
#     dexpert_states_ptr = DEXPERT_STATES + offs_m[:, None] * stride_dom
#     hidden_states_ptr = HIDDEN_STATES + offs_m[:, None] * stride_hm
#     indices_ptr = INDICES + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
#     routing_weights_ptr = (
#         ROUTING_WEIGHTS + offs_m[:, None] * stride_rm + offs_k[None, :] * stride_rk
#     )
#     expert_scores_ptr = (
#         EXPERT_SCORES + offs_m[:, None] * stride_sm + offs_k[None, :] * stride_sk
#     )
#     dhidden_states_ptr = DHIDDEN_STATES + offs_m[:, None] * stride_dhm
#     drouting_weights_ptr = (
#         DROUTING_WEIGHTS + offs_m[:, None] * stride_drm + offs_k[None, :] * stride_drk
#     )

#     # Load indices
#     if EVEN_M:
#         indices = tl.load(indices_ptr)
#     else:
#         indices = tl.load(
#             indices_ptr,
#             mask=offs_m[:, None] < num_tokens,
#             other=0,
#         )

#     # Initialize weights pointers
#     down_weights_ptr = DOWN_WEIGHTS + indices[:, :, None] * stride_de
#     up_weights_ptr = UP_WEIGHTS + indices[:, :, None] * stride_ue
#     ddown_weights_ptr = DDOWN_WEIGHTS + indices[:, :, None] * stride_dde
#     dup_weights_ptr = DUP_WEIGHTS + indices[:, :, None] * stride_due

#     # Load expert scores
#     if EVEN_M:
#         expert_scores = tl.load(expert_scores_ptr)
#     else:
#         expert_scores = tl.load(
#             expert_scores_ptr,
#             mask=offs_m[:, None] < num_tokens,
#             other=0.0,
#         )

#     # Load routing weights
#     if EVEN_M:
#         routing_weights = tl.load(routing_weights_ptr)
#     else:
#         routing_weights = tl.load(
#             routing_weights_ptr,
#             mask=offs_m[:, None] < num_tokens,
#             other=0.0,
#         )

#     # Compute activation weights
#     act_weights = silu(expert_scores).cast(routing_weights.dtype)
#     # Compute gated weights
#     gated_weights = act_weights * routing_weights

#     # Initialize expert weights gradient accumulator
#     dexpert_weights = tl.zeros((BLOCK_M, num_experts_per_tok), dtype=tl.float32)

#     for start_n in range(0, hidden_size, BLOCK_N):
#         start_n = tl.multiple_of(start_n, BLOCK_N)
#         offs_n = offs_nb + start_n

#         # Load expert states gradients
#         if EVEN_M:
#             if EVEN_N:
#                 dexpert_states = tl.load(
#                     dexpert_states_ptr + offs_n[None, :] * stride_don
#                 )
#             else:
#                 dexpert_states = tl.load(
#                     dexpert_states_ptr + offs_n[None, :] * stride_don,
#                     mask=offs_n[None, :] < hidden_size,
#                     other=0.0,
#                 )
#         else:
#             if EVEN_N:
#                 dexpert_states = tl.load(
#                     dexpert_states_ptr + offs_n[None, :] * stride_don,
#                     mask=offs_m[:, None] < num_tokens,
#                     other=0.0,
#                 )
#             else:
#                 dexpert_states = tl.load(
#                     dexpert_states_ptr + offs_n[None, :] * stride_don,
#                     mask=(offs_m[:, None] < num_tokens)
#                     & (offs_n[None, :] < hidden_size),
#                     other=0.0,
#                 )

#         # Load up weights
#         if EVEN_M:
#             if EVEN_N:
#                 up_weights = tl.load(up_weights_ptr + offs_n[None, None, :] * stride_un)
#             else:
#                 up_weights = tl.load(
#                     up_weights_ptr + offs_n[None, None, :] * stride_un,
#                     mask=offs_n[None, None, :] < hidden_size,
#                     other=0.0,
#                 )
#         else:
#             if EVEN_N:
#                 up_weights = tl.load(
#                     up_weights_ptr + offs_n[None, None, :] * stride_un,
#                     mask=offs_m[:, None, None] < num_tokens,
#                     other=0.0,
#                 )
#             else:
#                 up_weights = tl.load(
#                     up_weights_ptr + offs_n[None, None, :] * stride_un,
#                     mask=(offs_m[:, None, None] < num_tokens)
#                     & (offs_n[None, None, :] < hidden_size),
#                     other=0.0,
#                 )

#         # Compute up weights gradient
#         dup_weights = gated_weights[:, :, None] * dexpert_states[:, None, :]

#         # Write back up weights gradient
#         if EVEN_M:
#             if EVEN_N:
#                 tl.atomic_add(
#                     dup_weights_ptr + offs_n[None, None, :] * stride_dun, dup_weights
#                 )
#             else:
#                 tl.atomic_add(
#                     dup_weights_ptr + offs_n[None, None, :] * stride_dun,
#                     dup_weights,
#                     mask=offs_n[None, None, :] < hidden_size,
#                 )
#         else:
#             if EVEN_N:
#                 tl.atomic_add(
#                     dup_weights_ptr + offs_n[None, None, :] * stride_dun,
#                     dup_weights,
#                     mask=offs_m[:, None, None] < num_tokens,
#                 )
#             else:
#                 tl.atomic_add(
#                     dup_weights_ptr + offs_n[None, None, :] * stride_dun,
#                     dup_weights,
#                     mask=(offs_m[:, None, None] < num_tokens)
#                     & (offs_n[None, None, :] < hidden_size),
#                 )

#         # Compute expert weights gradient
#         dexpert_weights += tl.sum(dexpert_states[:, None, :] * up_weights, axis=2)

#     # Compute routing weights gradient
#     drouting_weights = dexpert_weights * act_weights

#     # Write back routing weights gradient
#     if EVEN_M:
#         tl.store(drouting_weights_ptr, drouting_weights)
#     else:
#         tl.store(
#             drouting_weights_ptr,
#             drouting_weights,
#             mask=offs_m[:, None] < num_tokens,
#         )

#     # Compute expert scores gradient
#     act = tl.sigmoid(expert_scores)
#     dact = act + expert_scores * act * (1.0 - act)
#     dexpert_scores = dexpert_weights * routing_weights * dact

#     for start_n in range(0, hidden_size, BLOCK_N):
#         start_n = tl.multiple_of(start_n, BLOCK_N)
#         offs_n = offs_nb + start_n

#         # Load hidden states
#         if EVEN_M:
#             if EVEN_N:
#                 hidden_states = tl.load(hidden_states_ptr + offs_n[None, :] * stride_hn)
#             else:
#                 hidden_states = tl.load(
#                     hidden_states_ptr + offs_n[None, :] * stride_hn,
#                     mask=offs_n[None, :] < hidden_size,
#                     other=0.0,
#                 )
#         else:
#             if EVEN_N:
#                 hidden_states = tl.load(
#                     hidden_states_ptr + offs_n[None, :] * stride_hn,
#                     mask=offs_m[:, None] < num_tokens,
#                     other=0.0,
#                 )
#             else:
#                 hidden_states = tl.load(
#                     hidden_states_ptr + offs_n[None, :] * stride_hn,
#                     mask=(offs_m[:, None] < num_tokens)
#                     & (offs_n[None, :] < hidden_size),
#                     other=0.0,
#                 )

#         # Load down weights
#         if EVEN_M:
#             if EVEN_N:
#                 down_weights = tl.load(
#                     down_weights_ptr + offs_n[None, None, :] * stride_dn
#                 )
#             else:
#                 down_weights = tl.load(
#                     down_weights_ptr + offs_n[None, None, :] * stride_dn,
#                     mask=offs_n[None, None, :] < hidden_size,
#                     other=0.0,
#                 )
#         else:
#             if EVEN_N:
#                 down_weights = tl.load(
#                     down_weights_ptr + offs_n[None, None, :] * stride_dn,
#                     mask=offs_m[:, None, None] < num_tokens,
#                     other=0.0,
#                 )
#             else:
#                 down_weights = tl.load(
#                     down_weights_ptr + offs_n[None, None, :] * stride_dn,
#                     mask=(offs_m[:, None, None] < num_tokens)
#                     & (offs_n[None, None, :] < hidden_size),
#                     other=0.0,
#                 )

#         # Compute hidden states gradient
#         dhidden_states = tl.sum(dexpert_scores[:, :, None] * down_weights, axis=1)

#         # Write back hidden states gradient
#         if EVEN_M:
#             if EVEN_N:
#                 tl.store(
#                     dhidden_states_ptr + offs_n[None, :] * stride_dhn, dhidden_states
#                 )
#             else:
#                 tl.store(
#                     dhidden_states_ptr + offs_n[None, :] * stride_dhn,
#                     dhidden_states,
#                     mask=offs_n[None, :] < hidden_size,
#                 )
#         else:
#             if EVEN_N:
#                 tl.store(
#                     dhidden_states_ptr + offs_n[None, :] * stride_dhn,
#                     dhidden_states,
#                     mask=offs_m[:, None] < num_tokens,
#                 )
#             else:
#                 tl.store(
#                     dhidden_states_ptr + offs_n[None, :] * stride_dhn,
#                     dhidden_states,
#                     mask=(offs_m[:, None] < num_tokens)
#                     & (offs_n[None, :] < hidden_size),
#                 )

#         # Compute down weights gradient
#         ddown_weights = dexpert_scores[:, :, None] * hidden_states[:, None, :]

#         # Write back down weights gradient
#         if EVEN_M:
#             if EVEN_N:
#                 tl.atomic_add(
#                     ddown_weights_ptr + offs_n[None, None, :] * stride_ddn,
#                     ddown_weights,
#                 )
#             else:
#                 tl.atomic_add(
#                     ddown_weights_ptr + offs_n[None, None, :] * stride_ddn,
#                     ddown_weights,
#                     mask=offs_n[None, None, :] < hidden_size,
#                 )
#         else:
#             if EVEN_N:
#                 tl.atomic_add(
#                     ddown_weights_ptr + offs_n[None, None, :] * stride_ddn,
#                     ddown_weights,
#                     mask=offs_m[:, None, None] < num_tokens,
#                 )
#             else:
#                 tl.atomic_add(
#                     ddown_weights_ptr + offs_n[None, None, :] * stride_ddn,
#                     ddown_weights,
#                     mask=(offs_m[:, None, None] < num_tokens)
#                     & (offs_n[None, None, :] < hidden_size),
#                 )


@triton.autotune(
    configs=[
        # For num_experts_per_tok > 16
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=8, num_stages=4),
        # For num_experts_per_tok <= 16
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=4),
    ],
    key=["num_tokens", "hidden_size", "num_experts_per_tok"],
    reset_to_zero=["DUP_WEIGHTS", "DROUTING_WEIGHTS", "DEXPERT_SCORES"],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["num_tokens"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["hidden_size"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _bwd_states_kernel(
    DEXPERT_STATES,
    UP_WEIGHTS,
    INDICES,
    ROUTING_WEIGHTS,
    EXPERT_SCORES,
    DEXPERT_SCORES,
    DUP_WEIGHTS,
    DROUTING_WEIGHTS,
    stride_dom,
    stride_don,
    stride_ue,
    stride_un,
    stride_im,
    stride_ik,
    stride_rm,
    stride_rk,
    stride_sm,
    stride_sk,
    stride_dsm,
    stride_dsk,
    stride_due,
    stride_dun,
    stride_drm,
    stride_drk,
    num_tokens,
    hidden_size: tl.constexpr,
    num_experts_per_tok: tl.constexpr,
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
    offs_k = tl.arange(0, num_experts_per_tok)

    # Initialize pointers
    dexpert_states_ptr = (
        DEXPERT_STATES + offs_m[:, None] * stride_dom + offs_n[None, :] * stride_don
    )
    indices_ptr = INDICES + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
    routing_weights_ptr = (
        ROUTING_WEIGHTS + offs_m[:, None] * stride_rm + offs_k[None, :] * stride_rk
    )
    expert_scores_ptr = (
        EXPERT_SCORES + offs_m[:, None] * stride_sm + offs_k[None, :] * stride_sk
    )
    dexpert_scores_ptr = (
        DEXPERT_SCORES + offs_m[:, None] * stride_dsm + offs_k[None, :] * stride_dsk
    )
    drouting_weights_ptr = (
        DROUTING_WEIGHTS + offs_m[:, None] * stride_drm + offs_k[None, :] * stride_drk
    )

    # Load indices
    if EVEN_M:
        indices = tl.load(indices_ptr)
    else:
        indices = tl.load(
            indices_ptr,
            mask=offs_m[:, None] < num_tokens,
            other=0,
        )

    # Initialize weights pointers
    up_weights_ptr = (
        UP_WEIGHTS + indices[:, :, None] * stride_ue + offs_n[None, None, :] * stride_un
    )
    dup_weights_ptr = (
        DUP_WEIGHTS
        + indices[:, :, None] * stride_due
        + offs_n[None, None, :] * stride_dun
    )

    # Load expert scores
    if EVEN_M:
        expert_scores = tl.load(expert_scores_ptr)
    else:
        expert_scores = tl.load(
            expert_scores_ptr,
            mask=offs_m[:, None] < num_tokens,
            other=0.0,
        )

    # Load dexpert states
    if EVEN_M:
        if EVEN_N:
            dexpert_states = tl.load(dexpert_states_ptr)
        else:
            dexpert_states = tl.load(
                dexpert_states_ptr,
                mask=offs_n[None, :] < hidden_size,
                other=0.0,
            )
    else:
        if EVEN_N:
            dexpert_states = tl.load(
                dexpert_states_ptr,
                mask=offs_m[:, None] < num_tokens,
                other=0.0,
            )
        else:
            dexpert_states = tl.load(
                dexpert_states_ptr,
                mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < hidden_size),
                other=0.0,
            )

    # Load routing weights
    if EVEN_M:
        routing_weights = tl.load(routing_weights_ptr)
    else:
        routing_weights = tl.load(
            routing_weights_ptr,
            mask=offs_m[:, None] < num_tokens,
            other=0.0,
        )

    # Compute activation weights
    act_weights = silu(expert_scores).cast(dexpert_states.dtype)

    # Compute up weights gradient
    gated_weights = act_weights * routing_weights
    dup_weights = gated_weights[:, :, None] * dexpert_states[:, None, :]

    # Write back up weights gradient
    if EVEN_M:
        if EVEN_N:
            tl.atomic_add(dup_weights_ptr, dup_weights)
        else:
            tl.atomic_add(
                dup_weights_ptr,
                dup_weights,
                mask=offs_n[None, None, :] < hidden_size,
            )
    else:
        if EVEN_N:
            tl.atomic_add(
                dup_weights_ptr,
                dup_weights,
                mask=offs_m[:, None, None] < num_tokens,
            )
        else:
            tl.atomic_add(
                dup_weights_ptr,
                dup_weights,
                mask=(offs_m[:, None, None] < num_tokens)
                & (offs_n[None, None, :] < hidden_size),
            )

    # Load up weights
    if EVEN_M:
        if EVEN_N:
            up_weights = tl.load(up_weights_ptr)
        else:
            up_weights = tl.load(
                up_weights_ptr,
                mask=offs_n[None, None, :] < hidden_size,
                other=0.0,
            )
    else:
        if EVEN_N:
            up_weights = tl.load(
                up_weights_ptr,
                mask=offs_m[:, None, None] < num_tokens,
                other=0.0,
            )
        else:
            up_weights = tl.load(
                up_weights_ptr,
                mask=(offs_m[:, None, None] < num_tokens)
                & (offs_n[None, None, :] < hidden_size),
                other=0.0,
            )

    # Compute expert weights gradient
    dexpert_weights = tl.sum(dexpert_states[:, None, :] * up_weights, axis=2)

    # Compute expert scores gradient
    act = tl.sigmoid(expert_scores)
    dact = act + expert_scores * act * (1.0 - act)
    dexpert_scores = dexpert_weights * routing_weights * dact

    # Compute routing weights gradient
    drouting_weights = dexpert_weights * act_weights

    # Write back dexpert scores gradient
    if EVEN_M:
        tl.atomic_add(dexpert_scores_ptr, dexpert_scores)
    else:
        tl.atomic_add(
            dexpert_scores_ptr,
            dexpert_scores,
            mask=offs_m[:, None] < num_tokens,
        )

    # Write back routing weights gradient
    if EVEN_M:
        tl.atomic_add(drouting_weights_ptr, drouting_weights)
    else:
        tl.atomic_add(
            drouting_weights_ptr,
            drouting_weights,
            mask=offs_m[:, None] < num_tokens,
        )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=8, num_stages=4),
    ],
    key=["num_tokens", "hidden_size", "num_experts_per_tok"],
    reset_to_zero=["DDOWN_WEIGHTS"],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["num_tokens"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["hidden_size"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _bwd_scores_kernel(
    DEXPERT_SCORES,
    HIDDEN_STATES,
    DOWN_WEIGHTS,
    INDICES,
    DHIDDEN_STATES,
    DDOWN_WEIGHTS,
    stride_dsm,
    stride_dsk,
    stride_hm,
    stride_hn,
    stride_de,
    stride_dn,
    stride_im,
    stride_ik,
    stride_dhm,
    stride_dhn,
    stride_dde,
    stride_ddn,
    num_tokens,
    hidden_size: tl.constexpr,
    num_experts_per_tok: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    # Initialize offsets
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, num_experts_per_tok)

    # Initialize pointers
    dexpert_scores_ptr = (
        DEXPERT_SCORES + offs_m[:, None] * stride_dsm + offs_k[None, :] * stride_dsk
    )
    indices_ptr = INDICES + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
    hidden_states_ptr = HIDDEN_STATES + offs_m[:, None] * stride_hm
    dhidden_states_ptr = DHIDDEN_STATES + offs_m[:, None] * stride_dhm

    # Load indices
    if EVEN_M:
        indices = tl.load(indices_ptr)
    else:
        indices = tl.load(
            indices_ptr,
            mask=offs_m[:, None] < num_tokens,
            other=0,
        )

    # Initialize weights pointers
    down_weights_ptr = DOWN_WEIGHTS + indices[:, :, None] * stride_de
    ddown_weights_ptr = DDOWN_WEIGHTS + indices[:, :, None] * stride_dde

    # Load dexpert scores
    if EVEN_M:
        dexpert_scores = tl.load(dexpert_scores_ptr)
    else:
        dexpert_scores = tl.load(
            dexpert_scores_ptr,
            mask=offs_m[:, None] < num_tokens,
            other=0.0,
        )

    # Loop over hidden dimension
    for start_n in range(0, hidden_size, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_nb = offs_n + start_n

        # Load down weights
        if EVEN_M:
            if EVEN_N:
                down_weights = tl.load(
                    down_weights_ptr + offs_nb[None, None, :] * stride_dn
                )
            else:
                down_weights = tl.load(
                    down_weights_ptr + offs_nb[None, None, :] * stride_dn,
                    mask=offs_nb[None, None, :] < hidden_size,
                    other=0.0,
                )
        else:
            if EVEN_N:
                down_weights = tl.load(
                    down_weights_ptr + offs_nb[None, None, :] * stride_dn,
                    mask=offs_m[:, None, None] < num_tokens,
                    other=0.0,
                )
            else:
                down_weights = tl.load(
                    down_weights_ptr + offs_nb[None, None, :] * stride_dn,
                    mask=(offs_m[:, None, None] < num_tokens)
                    & (offs_nb[None, None, :] < hidden_size),
                    other=0.0,
                )

        # Compute hidden states gradient
        dhidden_states = tl.sum(dexpert_scores[:, :, None] * down_weights, axis=1)

        # Write back hidden states gradient
        if EVEN_M:
            if EVEN_N:
                tl.store(
                    dhidden_states_ptr + offs_nb[None, :] * stride_dhn, dhidden_states
                )
            else:
                tl.store(
                    dhidden_states_ptr + offs_nb[None, :] * stride_dhn,
                    dhidden_states,
                    mask=offs_nb[None, :] < hidden_size,
                )
        else:
            if EVEN_N:
                tl.store(
                    dhidden_states_ptr + offs_nb[None, :] * stride_dhn,
                    dhidden_states,
                    mask=offs_m[:, None] < num_tokens,
                )
            else:
                tl.store(
                    dhidden_states_ptr + offs_nb[None, :] * stride_dhn,
                    dhidden_states,
                    mask=(offs_m[:, None] < num_tokens)
                    & (offs_nb[None, :] < hidden_size),
                )

        # Load hidden states
        if EVEN_M:
            if EVEN_N:
                hidden_states = tl.load(
                    hidden_states_ptr + offs_nb[None, :] * stride_hn
                )
            else:
                hidden_states = tl.load(
                    hidden_states_ptr + offs_nb[None, :] * stride_hn,
                    mask=offs_nb[None, :] < hidden_size,
                    other=0.0,
                )
        else:
            if EVEN_N:
                hidden_states = tl.load(
                    hidden_states_ptr + offs_nb[None, :] * stride_hn,
                    mask=offs_m[:, None] < num_tokens,
                    other=0.0,
                )
            else:
                hidden_states = tl.load(
                    hidden_states_ptr + offs_nb[None, :] * stride_hn,
                    mask=(offs_m[:, None] < num_tokens)
                    & (offs_nb[None, :] < hidden_size),
                    other=0.0,
                )

        # Compute down weights gradient
        ddown_weights = dexpert_scores[:, :, None] * hidden_states[:, None, :]

        # Write back down weights gradient
        if EVEN_M:
            if EVEN_N:
                tl.atomic_add(
                    ddown_weights_ptr + offs_nb[None, None, :] * stride_ddn,
                    ddown_weights,
                )
            else:
                tl.atomic_add(
                    ddown_weights_ptr + offs_nb[None, None, :] * stride_ddn,
                    ddown_weights,
                    mask=offs_nb[None, None, :] < hidden_size,
                )
        else:
            if EVEN_N:
                tl.atomic_add(
                    ddown_weights_ptr + offs_nb[None, None, :] * stride_ddn,
                    ddown_weights,
                    mask=offs_m[:, None, None] < num_tokens,
                )
            else:
                tl.atomic_add(
                    ddown_weights_ptr + offs_nb[None, None, :] * stride_ddn,
                    ddown_weights,
                    mask=(offs_m[:, None, None] < num_tokens)
                    & (offs_nb[None, None, :] < hidden_size),
                )


def _flash_expert_forward(
    hidden_states: torch.Tensor,
    down_weights: torch.Tensor,
    up_weights: torch.Tensor,
    indices: torch.Tensor,
    routing_weights: torch.Tensor,
):
    num_tokens, hidden_size = hidden_states.shape
    num_experts_per_tok = routing_weights.shape[1]

    # Storage experts scores for backward
    expert_scores = torch.empty(
        (num_tokens, num_experts_per_tok),
        device=hidden_states.device,
        dtype=torch.float32,
    )
    expert_states = torch.empty_like(hidden_states)

    # Loop over tokens to compute expert scores
    # Just gather and matmul over hidden dimension, memory access is friendly for Tensor Cores
    def grid(META):
        return (triton.cdiv(num_tokens, META["BLOCK_M"]),)

    _fwd_scores_kernel[grid](
        hidden_states,
        down_weights,
        indices,
        expert_scores,
        hidden_states.stride(0),
        hidden_states.stride(1),
        down_weights.stride(0),
        down_weights.stride(1),
        indices.stride(0),
        indices.stride(1),
        expert_scores.stride(0),
        expert_scores.stride(1),
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        num_experts_per_tok=num_experts_per_tok,
    )

    # Loop over tokens and hidden dimension to compute expert states
    def grid(META):
        return (
            triton.cdiv(num_tokens, META["BLOCK_M"]),
            triton.cdiv(hidden_size, META["BLOCK_N"]),
        )

    _fwd_states_kernel[grid](
        expert_scores,
        up_weights,
        indices,
        routing_weights,
        expert_states,
        expert_scores.stride(0),
        expert_scores.stride(1),
        up_weights.stride(0),
        up_weights.stride(1),
        indices.stride(0),
        indices.stride(1),
        routing_weights.stride(0),
        routing_weights.stride(1),
        expert_states.stride(0),
        expert_states.stride(1),
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        num_experts_per_tok=num_experts_per_tok,
    )

    return expert_states, expert_scores


# This version is about 20% slower but saves about 10% memory, kept as backup
# def _flash_expert_backward(
#     dexpert_states: torch.Tensor,
#     hidden_states: torch.Tensor,
#     down_weights: torch.Tensor,
#     up_weights: torch.Tensor,
#     indices: torch.Tensor,
#     routing_weights: torch.Tensor,
#     expert_scores: torch.Tensor,
# ):
#     num_tokens, hidden_size = hidden_states.shape
#     num_experts_per_tok = routing_weights.shape[1]

#     dhidden_states = torch.empty_like(hidden_states)
#     ddown_weights = torch.empty_like(down_weights)
#     dup_weights = torch.empty_like(up_weights)
#     drouting_weights = torch.empty_like(routing_weights)

#     def grid(META):
#         return (triton.cdiv(num_tokens, META["BLOCK_M"]),)

#     _bwd_kernel[grid](
#         dexpert_states,
#         hidden_states,
#         down_weights,
#         up_weights,
#         indices,
#         routing_weights,
#         expert_scores,
#         dhidden_states,
#         ddown_weights,
#         dup_weights,
#         drouting_weights,
#         dexpert_states.stride(0),
#         dexpert_states.stride(1),
#         hidden_states.stride(0),
#         hidden_states.stride(1),
#         down_weights.stride(0),
#         down_weights.stride(1),
#         up_weights.stride(0),
#         up_weights.stride(1),
#         indices.stride(0),
#         indices.stride(1),
#         routing_weights.stride(0),
#         routing_weights.stride(1),
#         expert_scores.stride(0),
#         expert_scores.stride(1),
#         dhidden_states.stride(0),
#         dhidden_states.stride(1),
#         ddown_weights.stride(0),
#         ddown_weights.stride(1),
#         dup_weights.stride(0),
#         dup_weights.stride(1),
#         drouting_weights.stride(0),
#         drouting_weights.stride(1),
#         num_tokens=num_tokens,
#         hidden_size=hidden_size,
#         num_experts_per_tok=num_experts_per_tok,
#     )

#     return dhidden_states, ddown_weights, dup_weights, drouting_weights


def _flash_expert_backward(
    dexpert_states: torch.Tensor,
    hidden_states: torch.Tensor,
    down_weights: torch.Tensor,
    up_weights: torch.Tensor,
    indices: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_scores: torch.Tensor,
):
    num_tokens, hidden_size = hidden_states.shape
    num_experts_per_tok = routing_weights.shape[1]

    dhidden_states = torch.empty_like(hidden_states)
    ddown_weights = torch.empty_like(down_weights)
    dup_weights = torch.empty_like(up_weights)
    drouting_weights = torch.empty_like(routing_weights)
    dexpert_scores = torch.empty_like(expert_scores)

    def grid(META):
        return (
            triton.cdiv(num_tokens, META["BLOCK_M"]),
            triton.cdiv(hidden_size, META["BLOCK_N"]),
        )

    _bwd_states_kernel[grid](
        dexpert_states,
        up_weights,
        indices,
        routing_weights,
        expert_scores,
        dexpert_scores,
        dup_weights,
        drouting_weights,
        dexpert_states.stride(0),
        dexpert_states.stride(1),
        up_weights.stride(0),
        up_weights.stride(1),
        indices.stride(0),
        indices.stride(1),
        routing_weights.stride(0),
        routing_weights.stride(1),
        expert_scores.stride(0),
        expert_scores.stride(1),
        dexpert_scores.stride(0),
        dexpert_scores.stride(1),
        dup_weights.stride(0),
        dup_weights.stride(1),
        drouting_weights.stride(0),
        drouting_weights.stride(1),
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        num_experts_per_tok=num_experts_per_tok,
    )

    def grid(META):
        return (triton.cdiv(num_tokens, META["BLOCK_M"]),)

    _bwd_scores_kernel[grid](
        dexpert_scores,
        hidden_states,
        down_weights,
        indices,
        dhidden_states,
        ddown_weights,
        dexpert_scores.stride(0),
        dexpert_scores.stride(1),
        hidden_states.stride(0),
        hidden_states.stride(1),
        down_weights.stride(0),
        down_weights.stride(1),
        indices.stride(0),
        indices.stride(1),
        dhidden_states.stride(0),
        dhidden_states.stride(1),
        ddown_weights.stride(0),
        ddown_weights.stride(1),
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        num_experts_per_tok=num_experts_per_tok,
    )

    return dhidden_states, ddown_weights, dup_weights, drouting_weights


class FlashExpertFunc(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, hidden_states, down_weights, up_weights, indices, routing_weights):
        """
        Args:
            hidden_states: (batch_size * seq_len, hidden_size)
            down_weights: (num_experts, hidden_size)
            up_weights: (num_experts, hidden_size)
            indices: (batch_size * seq_len, num_experts_per_tok)
            routing_weights: (batch_size * seq_len, num_experts_per_tok)
        """

        experts_states, expert_scores = _flash_expert_forward(
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
            indices,
            routing_weights,
            expert_scores,
        )

        return experts_states

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dexperts_states):
        (
            hidden_states,
            down_weights,
            up_weights,
            indices,
            routing_weights,
            expert_scores,
        ) = ctx.saved_tensors

        dhidden_states, ddown_weights, dup_weights, drouting_weights = (
            _flash_expert_backward(
                dexperts_states,
                hidden_states,
                down_weights,
                up_weights,
                indices,
                routing_weights,
                expert_scores,
            )
        )

        return dhidden_states, ddown_weights, dup_weights, None, drouting_weights


def triton_flash_expert_func(
    hidden_states, down_weights, up_weights, indices, routing_weights
):
    return FlashExpertFunc.apply(
        hidden_states, down_weights, up_weights, indices, routing_weights
    )
