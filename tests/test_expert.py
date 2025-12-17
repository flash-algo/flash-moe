import pytest
import torch

import testing
from flash_moe.ops.flash_expert import triton_flash_expert_func


def pytorch_expert_forward(
    hidden_states: torch.Tensor,
    down_embed: torch.Tensor,
    up_embed: torch.Tensor,
    indices: torch.Tensor,
    routing_weights: torch.Tensor,
):
    num_tokens, _ = hidden_states.shape

    down_weights = down_embed[indices]
    up_weights = up_embed[indices]

    expert_weights = torch.matmul(down_weights, hidden_states.unsqueeze(-1)).view(
        num_tokens, -1
    )
    expert_weights = torch.nn.functional.silu(expert_weights) * routing_weights
    expert_states = torch.matmul(expert_weights.unsqueeze(-2), up_weights).squeeze(-2)
    return expert_states


def triton_expert_forward(
    hidden_states: torch.Tensor,
    down_embed: torch.Tensor,
    up_embed: torch.Tensor,
    indices: torch.Tensor,
    routing_weights: torch.Tensor,
):
    return triton_flash_expert_func(
        hidden_states, down_embed, up_embed, indices, routing_weights
    )


def pytorch_expert_backward(
    loss: torch.Tensor,
    hidden: torch.Tensor,
    down: torch.Tensor,
    up: torch.Tensor,
    routing: torch.Tensor,
):
    loss.backward()
    return (
        hidden.grad,
        down.grad,
        up.grad,
        routing.grad,
    )


def triton_expert_backward(
    loss: torch.Tensor,
    hidden: torch.Tensor,
    down: torch.Tensor,
    up: torch.Tensor,
    routing: torch.Tensor,
):
    loss.backward()
    return (
        hidden.grad,
        down.grad,
        up.grad,
        routing.grad,
    )


def make_forward_factory(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    device: torch.device,
    dtype: torch.dtype,
):
    gen = torch.Generator(device=device).manual_seed(0)
    hidden_states = torch.randn(
        num_tokens, hidden_size, device=device, dtype=dtype, generator=gen
    )
    down_embed = torch.randn(
        num_experts, hidden_size, device=device, dtype=dtype, generator=gen
    )
    up_embed = torch.randn(
        num_experts, hidden_size, device=device, dtype=dtype, generator=gen
    )
    indices = torch.randint(
        0, num_experts, (num_tokens, top_k), device=device, dtype=torch.int64
    )
    routing_weights = torch.randn(
        num_tokens, top_k, device=device, dtype=dtype, generator=gen
    )

    def _factory(_impl: testing.Implementation):
        args = (
            hidden_states.clone(),
            down_embed.clone(),
            up_embed.clone(),
            indices.clone(),
            routing_weights.clone(),
        )
        kwargs = {}
        return args, kwargs

    return _factory


def make_backward_factory(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    device: torch.device,
    dtype: torch.dtype,
):
    gen = torch.Generator(device=device).manual_seed(0)
    hidden_states = torch.randn(
        num_tokens, hidden_size, device=device, dtype=dtype, generator=gen
    )
    down_embed = torch.randn(
        num_experts, hidden_size, device=device, dtype=dtype, generator=gen
    )
    up_embed = torch.randn(
        num_experts, hidden_size, device=device, dtype=dtype, generator=gen
    )
    indices = torch.randint(
        0, num_experts, (num_tokens, top_k), device=device, dtype=torch.int64
    )
    routing_weights = torch.randn(
        num_tokens, top_k, device=device, dtype=dtype, generator=gen
    )

    def _factory(impl: testing.Implementation):
        idx = indices.clone()

        hidden = hidden_states.clone().detach().requires_grad_(True)
        down = down_embed.clone().detach().requires_grad_(True)
        up = up_embed.clone().detach().requires_grad_(True)
        routing = routing_weights.clone().detach().requires_grad_(True)

        if impl.backend == testing.Backend.PYTORCH:
            loss = pytorch_expert_forward(hidden, down, up, idx, routing).sum()
        elif impl.backend == testing.Backend.TRITON:
            loss = triton_expert_forward(hidden, down, up, idx, routing).sum()

        return (loss, hidden, down, up, routing), {}

    return _factory


@pytest.mark.parametrize(
    "dtype",
    [torch.float32],
)
@pytest.mark.parametrize(
    "case",
    [
        # (num_tokens, hidden_size, num_experts, top_k)
        (4096, 1024, 4096, 4),
        (4096, 1024, 4096, 8),
        (4096, 4096, 16384, 16),
        (4096, 4096, 16384, 32),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_expert_forward_throughput(
    dtype: torch.dtype, case: tuple[int, int, int, int]
) -> None:
    num_tokens, hidden_size, num_experts, top_k = case
    device = torch.device("cuda")

    print(
        f"[expert forward] num_tokens={num_tokens}, hidden_size={hidden_size}, num_experts={num_experts}, top_k={top_k}"
    )

    impls = testing.get_impls(
        pytorch_impl=pytorch_expert_forward,
        triton_impl=triton_expert_forward,
    )
    flops = 4.0 * num_tokens * top_k * hidden_size
    config = testing.BenchmarkConfig(warmup=5, repeat=1_000)
    results = testing.run_benchmarks(
        impls,
        make_forward_factory(
            num_tokens, hidden_size, num_experts, top_k, device, dtype
        ),
        flops=flops,
        config=config,
    )

    testing.show_benchmarks(results)


@pytest.mark.parametrize(
    "dtype",
    [torch.float32],
)
@pytest.mark.parametrize(
    "case",
    [
        # (num_tokens, hidden_size, num_experts, top_k)
        (4096, 1024, 4096, 4),
        (4096, 1024, 4096, 8),
        (4096, 4096, 16384, 16),
        (4096, 4096, 16384, 32),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_expert_backward_throughput(
    dtype: torch.dtype, case: tuple[int, int, int, int]
) -> None:
    num_tokens, hidden_size, num_experts, top_k = case
    device = torch.device("cuda")

    print(
        f"[expert backward] num_tokens={num_tokens}, hidden_size={hidden_size}, num_experts={num_experts}, top_k={top_k}"
    )

    impls = testing.get_impls(
        pytorch_impl=pytorch_expert_backward,
        triton_impl=triton_expert_backward,
    )
    flops = 2 * 4.0 * num_tokens * top_k * hidden_size
    config = testing.BenchmarkConfig(warmup=5, repeat=500)
    results = testing.run_benchmarks(
        impls,
        make_backward_factory(
            num_tokens, hidden_size, num_experts, top_k, device, dtype
        ),
        flops=flops,
        config=config,
    )

    testing.show_benchmarks(results)
