import pytest
import torch

import testing
from flash_moe.ops.triton.flash_expert_ec import triton_flash_expert_func
from flash_moe.ops.cutile.flash_expert import cutile_flash_expert_func


def pytorch_expert_forward(
    hidden_states: torch.Tensor,
    down_embed: torch.Tensor,
    up_embed: torch.Tensor,
    indices: torch.Tensor,
    routing_weights: torch.Tensor,
    shared_gate_proj: torch.Tensor,
    shared_up_proj: torch.Tensor,
    shared_down_proj: torch.Tensor,
):
    num_tokens, _ = hidden_states.shape

    down_weights = down_embed[indices]
    up_weights = up_embed[indices]

    expert_weights = torch.matmul(down_weights, hidden_states.unsqueeze(-1)).view(
        num_tokens, -1
    )
    expert_weights = torch.nn.functional.silu(expert_weights) * routing_weights
    expert_states = torch.matmul(expert_weights.unsqueeze(-2), up_weights).squeeze(-2)
    hidden_states = torch.matmul(
        torch.nn.functional.silu(torch.matmul(hidden_states, shared_gate_proj.t()))
        * torch.matmul(hidden_states, shared_up_proj.t()),
        shared_down_proj.t(),
    )
    hidden_states = hidden_states + expert_states
    return hidden_states


def triton_expert_forward(
    hidden_states: torch.Tensor,
    down_embed: torch.Tensor,
    up_embed: torch.Tensor,
    indices: torch.Tensor,
    routing_weights: torch.Tensor,
    shared_gate_proj: torch.Tensor,
    shared_up_proj: torch.Tensor,
    shared_down_proj: torch.Tensor,
):
    expert_states = triton_flash_expert_func(
        hidden_states, down_embed, up_embed, indices, routing_weights
    )
    hidden_states = torch.matmul(
        torch.nn.functional.silu(torch.matmul(hidden_states, shared_gate_proj.t()))
        * torch.matmul(hidden_states, shared_up_proj.t()),
        shared_down_proj.t(),
    )
    hidden_states = hidden_states + expert_states
    return hidden_states


def cutile_expert_forward(
    hidden_states: torch.Tensor,
    down_embed: torch.Tensor,
    up_embed: torch.Tensor,
    indices: torch.Tensor,
    routing_weights: torch.Tensor,
    shared_gate_proj: torch.Tensor,
    shared_up_proj: torch.Tensor,
    shared_down_proj: torch.Tensor,
):
    expert_states = cutile_flash_expert_func(
        hidden_states, down_embed, up_embed, indices, routing_weights
    )
    hidden_states = torch.matmul(
        torch.nn.functional.silu(torch.matmul(hidden_states, shared_gate_proj.t()))
        * torch.matmul(hidden_states, shared_up_proj.t()),
        shared_down_proj.t(),
    )
    hidden_states = hidden_states + expert_states
    return hidden_states


def pytorch_expert_backward(
    loss: torch.Tensor,
    hidden_states: torch.Tensor,
    down_embed: torch.Tensor,
    up_embed: torch.Tensor,
    indices: torch.Tensor,
    routing_weights: torch.Tensor,
    shared_gate_proj: torch.Tensor,
    shared_up_proj: torch.Tensor,
    shared_down_proj: torch.Tensor,
):
    loss.backward()
    return (
        hidden_states.grad,
        down_embed.grad,
        up_embed.grad,
        routing_weights.grad,
        shared_gate_proj.grad,
        shared_up_proj.grad,
        shared_down_proj.grad,
    )


def triton_expert_backward(
    loss: torch.Tensor,
    hidden_states: torch.Tensor,
    down_embed: torch.Tensor,
    up_embed: torch.Tensor,
    indices: torch.Tensor,
    routing_weights: torch.Tensor,
    shared_gate_proj: torch.Tensor,
    shared_up_proj: torch.Tensor,
    shared_down_proj: torch.Tensor,
):
    loss.backward()
    return (
        hidden_states.grad,
        down_embed.grad,
        up_embed.grad,
        routing_weights.grad,
        shared_gate_proj.grad,
        shared_up_proj.grad,
        shared_down_proj.grad,
    )


def cutile_expert_backward(
    loss: torch.Tensor,
    hidden_states: torch.Tensor,
    down_embed: torch.Tensor,
    up_embed: torch.Tensor,
    indices: torch.Tensor,
    routing_weights: torch.Tensor,
    shared_gate_proj: torch.Tensor,
    shared_up_proj: torch.Tensor,
    shared_down_proj: torch.Tensor,
):
    loss.backward()
    return (
        hidden_states.grad,
        down_embed.grad,
        up_embed.grad,
        routing_weights.grad,
        shared_gate_proj.grad,
        shared_up_proj.grad,
        shared_down_proj.grad,
    )


def make_forward_factory(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    intermediate_size: int,
    device: torch.device,
    dtype: torch.dtype,
):
    gen = torch.Generator(device=device).manual_seed(0)
    hidden_states = torch.randn(
        num_tokens, hidden_size, device=device, dtype=dtype, generator=gen
    ).normal_(0, 0.5)
    down_embed = torch.randn(
        num_experts, hidden_size, device=device, dtype=dtype, generator=gen
    ).normal_(0, 0.1)
    up_embed = torch.randn(
        num_experts, hidden_size, device=device, dtype=dtype, generator=gen
    ).normal_(0, 0.1)
    indices = torch.randint(
        0, num_experts, (num_tokens, top_k), device=device, dtype=torch.int64
    )
    routing_weights = torch.randn(
        num_tokens, top_k, device=device, dtype=dtype, generator=gen
    ).normal_(0, 0.1)
    shared_gate_proj = torch.randn(
        intermediate_size, hidden_size, device=device, dtype=dtype, generator=gen
    ).normal_(0, 0.1)
    shared_up_proj = torch.randn(
        intermediate_size, hidden_size, device=device, dtype=dtype, generator=gen
    ).normal_(0, 0.1)
    shared_down_proj = torch.randn(
        hidden_size, intermediate_size, device=device, dtype=dtype, generator=gen
    ).normal_(0, 0.1)

    def _factory(_impl: testing.Implementation):
        args = (
            hidden_states.clone(),
            down_embed.clone(),
            up_embed.clone(),
            indices.clone(),
            routing_weights.clone(),
            shared_gate_proj.clone(),
            shared_up_proj.clone(),
            shared_down_proj.clone(),
        )
        kwargs = {}
        return args, kwargs

    return _factory


def make_backward_factory(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    intermediate_size: int,
    device: torch.device,
    dtype: torch.dtype,
):
    gen = torch.Generator(device=device).manual_seed(0)
    hidden_states = torch.randn(
        num_tokens, hidden_size, device=device, dtype=dtype, generator=gen
    ).normal_(0, 0.5)
    down_embed = torch.randn(
        num_experts, hidden_size, device=device, dtype=dtype, generator=gen
    ).normal_(0, 0.1)
    up_embed = torch.randn(
        num_experts, hidden_size, device=device, dtype=dtype, generator=gen
    ).normal_(0, 0.1)
    indices = torch.randint(
        0, num_experts, (num_tokens, top_k), device=device, dtype=torch.int64
    )
    routing_weights = torch.randn(
        num_tokens, top_k, device=device, dtype=dtype, generator=gen
    ).normal_(0, 0.1)
    shared_gate_proj = torch.randn(
        intermediate_size, hidden_size, device=device, dtype=dtype, generator=gen
    ).normal_(0, 0.1)
    shared_up_proj = torch.randn(
        intermediate_size, hidden_size, device=device, dtype=dtype, generator=gen
    ).normal_(0, 0.1)
    shared_down_proj = torch.randn(
        hidden_size, intermediate_size, device=device, dtype=dtype, generator=gen
    ).normal_(0, 0.1)

    def _factory(impl: testing.Implementation):
        idx = indices.clone()

        hidden = hidden_states.clone().detach().requires_grad_(True)
        down = down_embed.clone().detach().requires_grad_(True)
        up = up_embed.clone().detach().requires_grad_(True)
        routing = routing_weights.clone().detach().requires_grad_(True)
        shared_gate = shared_gate_proj.clone().detach().requires_grad_(True)
        shared_up = shared_up_proj.clone().detach().requires_grad_(True)
        shared_down = shared_down_proj.clone().detach().requires_grad_(True)

        if impl.backend == testing.Backend.PYTORCH:
            loss = pytorch_expert_forward(
                hidden, down, up, idx, routing, shared_gate, shared_up, shared_down
            ).sum()
        elif impl.backend == testing.Backend.TRITON:
            loss = triton_expert_forward(
                hidden, down, up, idx, routing, shared_gate, shared_up, shared_down
            ).sum()

        return (
            loss,
            hidden,
            down,
            up,
            idx,
            routing,
            shared_gate,
            shared_up,
            shared_down,
        ), {}

    return _factory


@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16],
)
@pytest.mark.parametrize(
    "case",
    [
        # (num_tokens, hidden_size, num_experts, top_k, intermediate_size)
        (1024, 1024, 102400, 512, 1024),
        (1024, 1024, 102400, 1024, 1024),
        (1024, 1024, 102400, 2048, 1024),
        (1024, 1024, 102400, 4096, 1024),
        (2048, 1024, 102400, 512, 1024),
        (4096, 1024, 102400, 512, 1024),
        (8192, 1024, 102400, 512, 1024),
        (16384, 1024, 102400, 512, 1024),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_expert_forward_throughput(
    dtype: torch.dtype, case: tuple[int, int, int, int, int]
) -> None:
    num_tokens, hidden_size, num_experts, top_k, intermediate_size = case
    device = torch.device("cuda")

    print(
        f"[expert forward] num_tokens={num_tokens}, hidden_size={hidden_size}, num_experts={num_experts}, top_k={top_k}, intermediate_size={intermediate_size}"
    )

    impls = testing.get_impls(
        pytorch_impl=pytorch_expert_forward,
        triton_impl=triton_expert_forward,
        cutile_impl=cutile_expert_forward,
    )
    flops = (4.0 * num_tokens * top_k * hidden_size) + (
        6.0 * num_tokens * hidden_size * intermediate_size
    )
    config = testing.BenchmarkConfig(warmup=5, repeat=10)
    results = testing.run_benchmarks(
        impls,
        make_forward_factory(
            num_tokens,
            hidden_size,
            num_experts,
            top_k,
            intermediate_size,
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
        # (num_tokens, hidden_size, num_experts, top_k, intermediate_size)
        (1024, 1024, 102400, 512, 1024),
        (1024, 1024, 102400, 1024, 1024),
        (1024, 1024, 102400, 2048, 1024),
        (1024, 1024, 102400, 4096, 1024),
        (2048, 1024, 102400, 512, 1024),
        (4096, 1024, 102400, 512, 1024),
        (8192, 1024, 102400, 512, 1024),
        (16384, 1024, 102400, 512, 1024),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_expert_backward_throughput(
    dtype: torch.dtype, case: tuple[int, int, int, int, int]
) -> None:
    num_tokens, hidden_size, num_experts, top_k, intermediate_size = case
    device = torch.device("cuda")

    print(
        f"[expert backward] num_tokens={num_tokens}, hidden_size={hidden_size}, num_experts={num_experts}, top_k={top_k}, intermediate_size={intermediate_size}"
    )

    impls = testing.get_impls(
        pytorch_impl=pytorch_expert_backward,
        triton_impl=triton_expert_backward,
    )
    flops = 2 * 4.0 * num_tokens * top_k * hidden_size
    config = testing.BenchmarkConfig(warmup=5, repeat=10)
    results = testing.run_benchmarks(
        impls,
        make_backward_factory(
            num_tokens,
            hidden_size,
            num_experts,
            top_k,
            intermediate_size,
            device,
            dtype,
        ),
        flops=flops,
        config=config,
        validate=True,
    )

    testing.show_benchmarks(results)
