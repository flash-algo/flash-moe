import pytest
import torch

import testing
from flash_moe.ops.triton.gshared import triton_gshared_func
from flash_moe.ops.cutile.gshared import cutile_gshared_func


def pytorch_gshared_forward(
    hidden_states: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
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

    return final_hidden_states


def triton_gshared_forward(
    hidden_states: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    return triton_gshared_func(
        hidden_states,
        gate_proj,
        up_proj,
        down_proj,
        topk_weights,
        topk_ids,
    )


def cutile_gshared_forward(
    hidden_states: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    return cutile_gshared_func(
        hidden_states,
        gate_proj,
        up_proj,
        down_proj,
        topk_weights,
        topk_ids,
    )


def pytorch_gshared_backward(
    loss: torch.Tensor,
    hidden_states: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
):
    loss.backward()
    return (
        hidden_states.grad,
        gate_proj.grad,
        up_proj.grad,
        down_proj.grad,
        topk_weights.grad,
    )


def triton_gshared_backward(
    loss: torch.Tensor,
    hidden_states: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
):
    loss.backward()
    return (
        hidden_states.grad,
        gate_proj.grad,
        up_proj.grad,
        down_proj.grad,
        topk_weights.grad,
    )


def cutile_gshared_backward(
    loss: torch.Tensor,
    hidden_states: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
):
    loss.backward()
    return (
        hidden_states.grad,
        gate_proj.grad,
        up_proj.grad,
        down_proj.grad,
        topk_weights.grad,
    )


def make_forward_factory(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
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

    def factory(_impl: testing.Implementation):
        args = (
            hidden_states.clone(),
            gate_proj.clone(),
            up_proj.clone(),
            down_proj.clone(),
            topk_weights.clone(),
            topk_ids.clone(),
        )
        kwargs = {}
        return args, kwargs

    return factory


def make_backward_factory(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
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

    def factory(impl: testing.Implementation):
        hidden = hidden_states.clone().detach().requires_grad_(True)
        gate = gate_proj.clone().detach().requires_grad_(True)
        up = up_proj.clone().detach().requires_grad_(True)
        down = down_proj.clone().detach().requires_grad_(True)

        weights = topk_weights.clone().detach().requires_grad_(True)
        ids = topk_ids.clone().detach()

        if impl.backend == testing.Backend.PYTORCH:
            loss = pytorch_gshared_forward(hidden, gate, up, down, weights, ids).sum()
        elif impl.backend == testing.Backend.TRITON:
            loss = triton_gshared_forward(hidden, gate, up, down, weights, ids).sum()
        elif impl.backend == testing.Backend.CUTILE:
            loss = cutile_gshared_forward(hidden, gate, up, down, weights, ids).sum()
        else:
            raise ValueError(f"Unknown backend: {impl.backend}")

        return (loss, hidden, gate, up, down, weights, ids), {}

    return factory


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16],
)
@pytest.mark.parametrize(
    "case",
    [
        # num_tokens, hidden_size, intermediate_size, num_experts, topk
        (4096, 1024, 4096, 16, 2),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gshared_forward_throughput(
    dtype: torch.dtype, case: tuple[int, int, int, int, int]
) -> None:
    num_tokens, hidden_size, intermediate_size, num_experts, topk = case
    device = torch.device("cuda")

    print(
        f"[gshared forward] num_tokens={num_tokens}, hidden_size={hidden_size}, intermediate_size={intermediate_size}, num_experts={num_experts}, topk={topk}"
    )

    impls = testing.get_impls(
        pytorch_impl=pytorch_gshared_forward,
        triton_impl=triton_gshared_forward,
        cutile_impl=cutile_gshared_forward,
    )
    flops = 6.0 * num_tokens * hidden_size * intermediate_size * topk
    config = testing.BenchmarkConfig(warmup=5, repeat=10)
    results = testing.run_benchmarks(
        impls,
        make_forward_factory(
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


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16],
)
@pytest.mark.parametrize(
    "case",
    [
        # num_tokens, hidden_size, intermediate_size, num_experts, topk
        (4096, 1024, 4096, 16, 2),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(True, reason="Cutile gshared backward not implemented correctly")
def test_gshared_backward_throughput(
    dtype: torch.dtype, case: tuple[int, int, int, int, int]
) -> None:
    num_tokens, hidden_size, intermediate_size, num_experts, topk = case
    device = torch.device("cuda")

    print(
        f"[gshared backward] num_tokens={num_tokens}, hidden_size={hidden_size}, intermediate_size={intermediate_size}, num_experts={num_experts}, topk={topk}"
    )

    impls = testing.get_impls(
        pytorch_impl=pytorch_gshared_backward,
        triton_impl=triton_gshared_backward,
        cutile_impl=cutile_gshared_backward,
    )
    flops = 2.0 * 6.0 * num_tokens * hidden_size * intermediate_size * topk
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
