import pytest
import torch

import testing
from flash_moe.ops.triton.pkm import triton_pkm_func
from flash_moe.ops.cutile.pkm import cutile_pkm_func


def pytorch_pkm_forward(
    values: torch.Tensor,
    routing_weights: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    # Gather expert values
    gathered_values = values[indices]

    # Weighted sum
    output = (routing_weights.unsqueeze(-1) * gathered_values).sum(dim=1)

    return output


def triton_pkm_forward(
    values: torch.Tensor,
    routing_weights: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    return triton_pkm_func(
        values,
        routing_weights,
        indices,
    )


def cutile_pkm_forward(
    values: torch.Tensor,
    routing_weights: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    return cutile_pkm_func(
        values,
        routing_weights,
        indices,
    )


def pytorch_pkm_backward(
    loss: torch.Tensor,
    values: torch.Tensor,
    routing_weights: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    loss.backward()
    return (
        values.grad,
        routing_weights.grad,
    )


def triton_pkm_backward(
    loss: torch.Tensor,
    values: torch.Tensor,
    routing_weights: torch.Tensor,
    indices: torch.Tensor,
):
    loss.backward()
    return (
        values.grad,
        routing_weights.grad,
    )


def cutile_pkm_backward(
    loss: torch.Tensor,
    values: torch.Tensor,
    routing_weights: torch.Tensor,
    indices: torch.Tensor,
):
    loss.backward()
    return (
        values.grad,
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
    values = torch.randn(
        (num_experts, hidden_size),
        device=device,
        dtype=dtype,
        generator=gen,
    ).normal_(0, 0.1)

    # Generate random indices and routing weights
    indices = torch.stack(
        [torch.randperm(num_experts, device=device)[:topk] for _ in range(num_tokens)]
    )
    routing_weights = torch.softmax(
        torch.randn(num_tokens, topk, device=device), dim=-1
    ).to(dtype)

    def factory(_impl: testing.Implementation):
        args = (
            values.clone(),
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
    values = torch.randn(
        (num_experts, hidden_size),
        device=device,
        dtype=dtype,
        generator=gen,
    ).normal_(0, 0.1)

    # Generate random indices and routing weights
    indices = torch.stack(
        [torch.randperm(num_experts, device=device)[:topk] for _ in range(num_tokens)]
    )
    routing_weights = torch.softmax(
        torch.randn(num_tokens, topk, device=device), dim=-1
    ).to(dtype)

    def factory(impl: testing.Implementation):
        v = values.clone().detach().requires_grad_(True)
        weights = routing_weights.clone().detach().requires_grad_(True)
        ids = indices.clone().detach()

        if impl.backend == testing.Backend.PYTORCH:
            loss = pytorch_pkm_forward(v, weights, ids).sum()
        elif impl.backend == testing.Backend.TRITON:
            loss = triton_pkm_forward(v, weights, ids).sum()
        elif impl.backend == testing.Backend.CUTILE:
            loss = cutile_pkm_forward(v, weights, ids).sum()
        else:
            raise ValueError(f"Unknown backend: {impl.backend}")

        return (loss, v, weights, ids), {}

    return factory


@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16],
)
@pytest.mark.parametrize(
    "case",
    [
        # num_tokens, hidden_size, num_experts, topk
        (1024, 1024, 197136, 2500),
        (1024, 1024, 197136, 5000),
        (1024, 1024, 197136, 12500),
        (1024, 1024, 197136, 25000),

        (2048, 1024, 197136, 2500),
        (4096, 1024, 197136, 2500),
        (8192, 1024, 197136, 2500),
        (16384, 1024, 197136, 2500),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pkm_forward_throughput(
    dtype: torch.dtype, case: tuple[int, int, int, int]
) -> None:
    num_tokens, hidden_size, num_experts, topk = case
    device = torch.device("cuda")

    print(
        f"[pkm forward] num_tokens={num_tokens}, hidden_size={hidden_size}, num_experts={num_experts}, topk={topk}"
    )

    impls = testing.get_impls(
        pytorch_impl=pytorch_pkm_forward,
        triton_impl=triton_pkm_forward,
        cutile_impl=cutile_pkm_forward,
    )
    flops = 2.0 * num_tokens * hidden_size * topk
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
@pytest.mark.skipif(True, reason="PKM backward not implemented yet")
def test_pkm_backward_throughput(
    dtype: torch.dtype, case: tuple[int, int, int, int]
) -> None:
    num_tokens, hidden_size, num_experts, topk = case
    device = torch.device("cuda")

    print(
        f"[pkm backward] num_tokens={num_tokens}, hidden_size={hidden_size}, num_experts={num_experts}, topk={topk}"
    )

    impls = testing.get_impls(
        pytorch_impl=pytorch_pkm_backward,
        triton_impl=triton_pkm_backward,
        cutile_impl=cutile_pkm_backward,
    )
    flops = 2.0 * 2.0 * num_tokens * hidden_size * topk
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
