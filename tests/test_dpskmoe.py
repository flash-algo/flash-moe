import pytest
import torch

import testing
from flash_moe.ops.triton.dpskmoe import triton_dpskmoe_func
from flash_moe.ops.cutile.dpskmoe import cutile_dpskmoe_func
from flash_moe.ops.triton.flash_mlp import triton_flash_mlp_func
from flash_moe.ops.cutile.flash_mlp import cutile_flash_mlp_func


def pytorch_dpskmoe_forward(
    hidden_states: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_gate_proj: torch.Tensor,
    shared_up_proj: torch.Tensor,
    shared_down_proj: torch.Tensor,
) -> torch.Tensor:
    num_experts = gate_proj.shape[0]
    final_hidden_states = torch.zeros_like(hidden_states)

    expert_mask = torch.nn.functional.one_hot(
        topk_ids, num_classes=num_experts
    ).permute(2, 1, 0)
    expert_usage = expert_mask.sum(dim=(-1, -2)) > 0
    active_expert_ids = expert_usage.nonzero().squeeze(-1)

    for expert_id in active_expert_ids:
        expert_gate = gate_proj[expert_id]
        expert_up = up_proj[expert_id]
        expert_down = down_proj[expert_id]

        matched_ks, matched_token_ids = torch.where(expert_mask[expert_id])
        matched_tokens = hidden_states[matched_token_ids]

        gate_output = matched_tokens @ expert_gate.T
        up_output = matched_tokens @ expert_up.T
        swiglu_output = torch.nn.functional.silu(gate_output) * up_output
        expert_output = swiglu_output @ expert_down.T

        routing_weights = topk_weights[matched_token_ids, matched_ks]
        weighted_output = expert_output * routing_weights.unsqueeze(-1)

        final_hidden_states.index_add_(
            0,
            matched_token_ids,
            weighted_output.to(hidden_states.dtype),
        )
    shared_gate_output = hidden_states @ shared_gate_proj.T
    shared_up_output = hidden_states @ shared_up_proj.T
    shared_swiglu_output = (
        torch.nn.functional.silu(shared_gate_output) * shared_up_output
    )
    shared_expert_output = shared_swiglu_output @ shared_down_proj.T
    final_hidden_states += shared_expert_output

    return final_hidden_states


def triton_dpskmoe_forward(
    hidden_states: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_gate_proj: torch.Tensor,
    shared_up_proj: torch.Tensor,
    shared_down_proj: torch.Tensor,
) -> torch.Tensor:
    expert_output = triton_dpskmoe_func(
        hidden_states,
        gate_proj,
        up_proj,
        down_proj,
        topk_weights,
        topk_ids,
    )
    shared_expert_output = triton_flash_mlp_func(
        hidden_states,
        shared_gate_proj,
        shared_up_proj,
        shared_down_proj,
    )
    return expert_output + shared_expert_output


def cutile_dpskmoe_forward(
    hidden_states: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_gate_proj: torch.Tensor,
    shared_up_proj: torch.Tensor,
    shared_down_proj: torch.Tensor,
) -> torch.Tensor:
    expert_output = cutile_dpskmoe_func(
        hidden_states,
        gate_proj,
        up_proj,
        down_proj,
        topk_weights,
        topk_ids,
    )
    shared_expert_output = cutile_flash_mlp_func(
        hidden_states,
        shared_gate_proj,
        shared_up_proj,
        shared_down_proj,
    )
    return expert_output + shared_expert_output


def pytorch_dpskmoe_backward(
    loss: torch.Tensor,
    hidden_states: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_gate_proj: torch.Tensor,
    shared_up_proj: torch.Tensor,
    shared_down_proj: torch.Tensor,
):
    loss.backward()
    return (
        hidden_states.grad,
        gate_proj.grad,
        up_proj.grad,
        down_proj.grad,
        topk_weights.grad,
        shared_gate_proj.grad,
        shared_up_proj.grad,
        shared_down_proj.grad,
    )


def triton_dpskmoe_backward(
    loss: torch.Tensor,
    hidden_states: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_gate_proj: torch.Tensor,
    shared_up_proj: torch.Tensor,
    shared_down_proj: torch.Tensor,
):
    loss.backward()
    return (
        hidden_states.grad,
        gate_proj.grad,
        up_proj.grad,
        down_proj.grad,
        topk_weights.grad,
        shared_gate_proj.grad,
        shared_up_proj.grad,
        shared_down_proj.grad,
    )


def cutile_dpskmoe_backward(
    loss: torch.Tensor,
    hidden_states: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_gate_proj: torch.Tensor,
    shared_up_proj: torch.Tensor,
    shared_down_proj: torch.Tensor,
):
    loss.backward()
    return (
        hidden_states.grad,
        gate_proj.grad,
        up_proj.grad,
        down_proj.grad,
        topk_weights.grad,
        shared_gate_proj.grad,
        shared_up_proj.grad,
        shared_down_proj.grad,
    )


def make_forward_factory(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    shared_intermediate_size: int,
    topk: int,
    device: torch.device,
    dtype: torch.dtype,
):
    gen = torch.Generator(device=device).manual_seed(42)
    hidden_states = torch.randn(
        (num_tokens, hidden_size), device=device, dtype=dtype, generator=gen
    ).normal_(0, 0.5)
    gate_proj = torch.randn(
        (num_experts, intermediate_size, hidden_size),
        device=device,
        dtype=dtype,
        generator=gen,
    ).normal_(0, 0.1)
    up_proj = torch.randn(
        (num_experts, intermediate_size, hidden_size),
        device=device,
        dtype=dtype,
        generator=gen,
    ).normal_(0, 0.1)
    down_proj = torch.randn(
        (num_experts, hidden_size, intermediate_size),
        device=device,
        dtype=dtype,
        generator=gen,
    ).normal_(0, 0.1)
    topk_ids = torch.stack(
        [torch.randperm(num_experts, device=device)[:topk] for _ in range(num_tokens)]
    )
    topk_weights = torch.softmax(
        torch.randn(num_tokens, topk, device=device), dim=-1
    ).to(dtype)
    shared_gate_proj = torch.randn(
        (shared_intermediate_size, hidden_size),
        device=device,
        dtype=dtype,
        generator=gen,
    ).normal_(0, 0.1)
    shared_up_proj = torch.randn(
        (shared_intermediate_size, hidden_size),
        device=device,
        dtype=dtype,
        generator=gen,
    ).normal_(0, 0.1)
    shared_down_proj = torch.randn(
        (hidden_size, shared_intermediate_size),
        device=device,
        dtype=dtype,
        generator=gen,
    ).normal_(0, 0.1)

    def factory(_impl: testing.Implementation):
        args = (
            hidden_states.clone(),
            gate_proj.clone(),
            up_proj.clone(),
            down_proj.clone(),
            topk_weights.clone(),
            topk_ids.clone(),
            shared_gate_proj.clone(),
            shared_up_proj.clone(),
            shared_down_proj.clone(),
        )
        kwargs = {}
        return args, kwargs

    return factory


def make_backward_factory(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    shared_intermediate_size: int,
    topk: int,
    device: torch.device,
    dtype: torch.dtype,
):
    gen = torch.Generator(device=device).manual_seed(42)
    hidden_states = torch.randn(
        (num_tokens, hidden_size), device=device, dtype=dtype, generator=gen
    ).normal_(0, 0.5)
    gate_proj = torch.randn(
        (num_experts, intermediate_size, hidden_size),
        device=device,
        dtype=dtype,
        generator=gen,
    ).normal_(0, 0.1)
    up_proj = torch.randn(
        (num_experts, intermediate_size, hidden_size),
        device=device,
        dtype=dtype,
        generator=gen,
    ).normal_(0, 0.1)
    down_proj = torch.randn(
        (num_experts, hidden_size, intermediate_size),
        device=device,
        dtype=dtype,
        generator=gen,
    ).normal_(0, 0.1)
    topk_ids = torch.stack(
        [torch.randperm(num_experts, device=device)[:topk] for _ in range(num_tokens)]
    )
    topk_weights = torch.softmax(
        torch.randn(num_tokens, topk, device=device), dim=-1
    ).to(dtype)
    shared_gate_proj = torch.randn(
        (shared_intermediate_size, hidden_size),
        device=device,
        dtype=dtype,
        generator=gen,
    ).normal_(0, 0.1)
    shared_up_proj = torch.randn(
        (shared_intermediate_size, hidden_size),
        device=device,
        dtype=dtype,
        generator=gen,
    ).normal_(0, 0.1)
    shared_down_proj = torch.randn(
        (hidden_size, shared_intermediate_size),
        device=device,
        dtype=dtype,
        generator=gen,
    ).normal_(0, 0.1)

    def factory(impl: testing.Implementation):
        hidden = hidden_states.clone().detach().requires_grad_(True)
        gate = gate_proj.clone().detach().requires_grad_(True)
        up = up_proj.clone().detach().requires_grad_(True)
        down = down_proj.clone().detach().requires_grad_(True)
        weights = topk_weights.clone().detach().requires_grad_(True)
        ids = topk_ids.clone().detach()
        shared_gate = shared_gate_proj.clone().detach().requires_grad_(True)
        shared_up = shared_up_proj.clone().detach().requires_grad_(True)
        shared_down = shared_down_proj.clone().detach().requires_grad_(True)

        if impl.backend == testing.Backend.PYTORCH:
            loss = pytorch_dpskmoe_forward(
                hidden,
                gate,
                up,
                down,
                weights,
                ids,
                shared_gate,
                shared_up,
                shared_down,
            ).sum()
        elif impl.backend == testing.Backend.TRITON:
            loss = triton_dpskmoe_forward(
                hidden,
                gate,
                up,
                down,
                weights,
                ids,
                shared_gate,
                shared_up,
                shared_down,
            ).sum()
        elif impl.backend == testing.Backend.CUTILE:
            loss = cutile_dpskmoe_forward(
                hidden,
                gate,
                up,
                down,
                weights,
                ids,
                shared_gate,
                shared_up,
                shared_down,
            ).sum()
        else:
            raise ValueError(f"Unknown backend: {impl.backend}")

        return (loss, hidden, gate, up, down, weights, ids), {}

    return factory


@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16],
)
@pytest.mark.parametrize(
    "case",
    [
        # num_tokens, hidden_size, intermediate_size, num_experts, shared_intermediate_size, topk
        (1024, 1024, 256, 256, 1024, 4),
        (1024, 1024, 256, 256, 1024, 8),
        (1024, 1024, 256, 256, 1024, 16),
        (1024, 1024, 256, 256, 1024, 32),

        (2048, 1024, 256, 256, 1024, 4),
        (4096, 1024, 256, 256, 1024, 4),
        (8192, 1024, 256, 256, 1024, 4),
        (16384, 1024, 256, 256, 1024, 4),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dpskmoe_forward_throughput(
    dtype: torch.dtype, case: tuple[int, int, int, int, int, int]
) -> None:
    (
        num_tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        shared_intermediate_size,
        topk,
    ) = case
    device = torch.device("cuda")

    print(
        f"[dpskmoe forward] num_tokens={num_tokens}, hidden_size={hidden_size}, intermediate_size={intermediate_size}, num_experts={num_experts}, shared_intermediate_size={shared_intermediate_size}, topk={topk}"
    )

    impls = testing.get_impls(
        pytorch_impl=pytorch_dpskmoe_forward,
        triton_impl=triton_dpskmoe_forward,
        cutile_impl=cutile_dpskmoe_forward,
    )
    flops = (
        6.0
        * num_tokens
        * (
            hidden_size * intermediate_size * topk
            + hidden_size * shared_intermediate_size
        )
    )
    config = testing.BenchmarkConfig(warmup=5, repeat=10)
    results = testing.run_benchmarks(
        impls,
        make_forward_factory(
            num_tokens,
            hidden_size,
            intermediate_size,
            num_experts,
            shared_intermediate_size,
            topk,
            device,
            dtype,
        ),
        flops=flops,
        config=config,
        validate=True,
    )

    testing.show_benchmarks(results)


@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16],
)
@pytest.mark.parametrize(
    "case",
    [
        # num_tokens, hidden_size, intermediate_size, num_experts, shared_intermediate_size, topk
        (4096, 1024, 4096, 16, 1024, 2),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(True, reason="Cutile dpskmoe backward not implemented correctly")
def test_dpskmoe_backward_throughput(
    dtype: torch.dtype, case: tuple[int, int, int, int, int, int]
) -> None:
    (
        num_tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        shared_intermediate_size,
        topk,
    ) = case
    device = torch.device("cuda")

    print(
        f"[dpskmoe backward] num_tokens={num_tokens}, hidden_size={hidden_size}, intermediate_size={intermediate_size}, num_experts={num_experts}, shared_intermediate_size={shared_intermediate_size}, topk={topk}"
    )

    impls = testing.get_impls(
        pytorch_impl=pytorch_dpskmoe_backward,
        triton_impl=triton_dpskmoe_backward,
        cutile_impl=cutile_dpskmoe_backward,
    )
    flops = (
        2.0
        * 6.0
        * num_tokens
        * (
            hidden_size * intermediate_size * topk
            + hidden_size * shared_intermediate_size
        )
    )
    config = testing.BenchmarkConfig(warmup=5, repeat=10)
    results = testing.run_benchmarks(
        impls,
        make_backward_factory(
            num_tokens,
            hidden_size,
            intermediate_size,
            num_experts,
            topk,
            device,
            dtype,
        ),
        flops=flops,
        config=config,
        validate=True,
    )

    testing.show_benchmarks(results)
