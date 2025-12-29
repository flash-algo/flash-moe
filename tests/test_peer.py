import pytest
import torch

import testing
from flash_moe.ops.triton.peer import triton_peer_func
from flash_moe.ops.cutile.peer import cutile_peer_func


def pytorch_peer_forward(
    hidden_states: torch.Tensor,
    down_embed: torch.Tensor,
    up_embed: torch.Tensor,
    routing_weights: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    down_weights = down_embed[indices]
    up_weights = up_embed[indices]

    hidden_states = (hidden_states.unsqueeze(1) * down_weights).sum(dim=-1)
    hidden_states = torch.nn.functional.gelu(hidden_states)
    hidden_states = hidden_states * routing_weights
    hidden_states = (hidden_states.unsqueeze(-1) * up_weights).sum(dim=1)

    return hidden_states


def triton_peer_forward(
    hidden_states: torch.Tensor,
    down_embed: torch.Tensor,
    up_embed: torch.Tensor,
    routing_weights: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    return triton_peer_func(
        hidden_states,
        down_embed,
        up_embed,
        routing_weights,
        indices,
    )


def cutile_peer_forward(
    hidden_states: torch.Tensor,
    down_embed: torch.Tensor,
    up_embed: torch.Tensor,
    routing_weights: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    return cutile_peer_func(
        hidden_states,
        down_embed,
        up_embed,
        routing_weights,
        indices,
    )


def pytorch_peer_backward(
    loss: torch.Tensor,
    hidden_states: torch.Tensor,
    down_embed: torch.Tensor,
    up_embed: torch.Tensor,
    routing_weights: torch.Tensor,
    indices: torch.Tensor,
) -> tuple:
    loss.backward()
    return (
        hidden_states.grad,
        down_embed.grad,
        up_embed.grad,
        routing_weights.grad,
    )


def triton_peer_backward(
    loss: torch.Tensor,
    hidden_states: torch.Tensor,
    down_embed: torch.Tensor,
    up_embed: torch.Tensor,
    routing_weights: torch.Tensor,
    indices: torch.Tensor,
) -> tuple:
    loss.backward()
    return (
        hidden_states.grad,
        down_embed.grad,
        up_embed.grad,
        routing_weights.grad,
    )


def cutile_peer_backward(
    loss: torch.Tensor,
    hidden_states: torch.Tensor,
    down_embed: torch.Tensor,
    up_embed: torch.Tensor,
    routing_weights: torch.Tensor,
    indices: torch.Tensor,
) -> tuple:
    loss.backward()
    return (
        hidden_states.grad,
        down_embed.grad,
        up_embed.grad,
        routing_weights.grad,
    )


def make_forward_factory(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
    device: torch.device,
    dtype: torch.dtype,
):
    gen = torch.Generator(device=device).manual_seed(42)
    hidden_states = torch.randn(
        num_tokens,
        hidden_size,
        device=device,
        dtype=dtype,
        generator=gen,
    ).normal_(0, 0.5)
    down_embed = torch.randn(
        num_experts,
        hidden_size,
        device=device,
        dtype=dtype,
        generator=gen,
    ).normal_(0, 0.1)
    up_embed = torch.randn(
        num_experts,
        hidden_size,
        device=device,
        dtype=dtype,
        generator=gen,
    ).normal_(0, 0.1)
    indices = torch.stack(
        [torch.randperm(num_experts, device=device)[:topk] for _ in range(num_tokens)]
    )
    # ReLU activation on scores (non-competing as in PEER paper)
    routing_weights = torch.relu(
        torch.randn(num_tokens, topk, device=device, dtype=dtype)
    )

    def factory(_impl: testing.Implementation):
        args = (
            hidden_states.clone(),
            down_embed.clone(),
            up_embed.clone(),
            routing_weights.clone(),
            indices.clone(),
        )
        kwargs = {}
        return args, kwargs

    return factory


def make_backward_factory(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
    device: torch.device,
    dtype: torch.dtype,
):
    gen = torch.Generator(device=device).manual_seed(42)
    hidden_states = torch.randn(
        num_tokens,
        hidden_size,
        device=device,
        dtype=dtype,
        generator=gen,
    ).normal_(0, 0.5)
    down_embed = torch.randn(
        num_experts,
        hidden_size,
        device=device,
        dtype=dtype,
        generator=gen,
    ).normal_(0, 0.1)
    up_embed = torch.randn(
        num_experts,
        hidden_size,
        device=device,
        dtype=dtype,
        generator=gen,
    ).normal_(0, 0.1)

    indices = torch.stack(
        [torch.randperm(num_experts, device=device)[:topk] for _ in range(num_tokens)]
    )
    routing_weights = torch.relu(
        torch.randn(num_tokens, topk, device=device, dtype=dtype)
    )

    def factory(impl: testing.Implementation):
        hidden = hidden_states.clone().detach().requires_grad_(True)
        down = down_embed.clone().detach().requires_grad_(True)
        up = up_embed.clone().detach().requires_grad_(True)
        weights = routing_weights.clone().detach().requires_grad_(True)
        ids = indices.clone().detach()

        if impl.backend == testing.Backend.PYTORCH:
            loss = pytorch_peer_forward(hidden, down, up, weights, ids).sum()
        elif impl.backend == testing.Backend.TRITON:
            loss = triton_peer_forward(hidden, down, up, weights, ids).sum()
        elif impl.backend == testing.Backend.CUTILE:
            loss = cutile_peer_forward(hidden, down, up, weights, ids).sum()
        else:
            raise ValueError(f"Unknown backend: {impl.backend}")

        return (loss, hidden, down, up, weights, ids), {}

    return factory


@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16],
)
@pytest.mark.parametrize(
    "case",
    [
        # num_tokens, hidden_size, num_experts, topk
        (1024, 1024, 102400, 1250),
        (1024, 1024, 102400, 2500),
        (1024, 1024, 102400, 6250),
        (1024, 1024, 102400, 12500),
        (2048, 1024, 102400, 1250),
        (4096, 1024, 102400, 1250),
        (8192, 1024, 102400, 1250),
        (16384, 1024, 102400, 1250),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_peer_forward_throughput(
    dtype: torch.dtype, case: tuple[int, int, int, int]
) -> None:
    num_tokens, hidden_size, num_experts, topk = case
    device = torch.device("cuda")

    print(
        f"[peer forward] num_tokens={num_tokens}, hidden_size={hidden_size}, num_experts={num_experts}, topk={topk}"
    )

    impls = testing.get_impls(
        pytorch_impl=pytorch_peer_forward,
        triton_impl=triton_peer_forward,
        cutile_impl=cutile_peer_forward,
    )
    flops = 4.0 * num_tokens * hidden_size * topk
    config = testing.BenchmarkConfig(warmup=5, repeat=10)
    results = testing.run_benchmarks(
        impls,
        make_forward_factory(
            num_tokens,
            hidden_size,
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
    [torch.bfloat16],
)
@pytest.mark.parametrize(
    "case",
    [
        # num_tokens, hidden_size, num_experts, topk
        (1024, 256, 16384, 16),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(True, reason="PEER backward not implemented yet")
def test_peer_backward_throughput(
    dtype: torch.dtype, case: tuple[int, int, int, int]
) -> None:
    num_tokens, hidden_size, num_experts, topk = case
    device = torch.device("cuda")

    print(
        f"[peer backward] num_tokens={num_tokens}, hidden_size={hidden_size}, num_experts={num_experts}, topk={topk}"
    )

    impls = testing.get_impls(
        pytorch_impl=pytorch_peer_backward,
        triton_impl=triton_peer_backward,
        cutile_impl=cutile_peer_backward,
    )
    flops = 2.0 * 4.0 * num_tokens * hidden_size * topk
    config = testing.BenchmarkConfig(warmup=5, repeat=10)
    results = testing.run_benchmarks(
        impls,
        make_backward_factory(
            num_tokens,
            hidden_size,
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
