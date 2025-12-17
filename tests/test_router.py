import pytest
import torch

import testing
from flash_moe.ops.flash_router import triton_flash_router_func


def pytorch_router_forward(router_logits: torch.Tensor, num_keys: int, top_k: int):
    (scores_x, scores_y), (indices_x, indices_y) = router_logits.topk(num_keys, dim=-1)
    all_scores = scores_x.unsqueeze(-1) + scores_y.unsqueeze(-2)
    all_indices = indices_x.unsqueeze(-1) * num_keys + indices_y.unsqueeze(-2)
    all_scores = all_scores.view(*all_scores.shape[:-2], -1)
    all_indices = all_indices.view(*all_indices.shape[:-2], -1)
    scores, pos_idx = all_scores.topk(top_k, dim=-1)
    _ = all_indices.gather(-1, pos_idx)
    return scores


def triton_router_forward(router_logits: torch.Tensor, num_keys: int, top_k: int):
    scores, _ = triton_flash_router_func(router_logits, num_keys, top_k)
    return scores


def pytorch_router_backward(
    loss: torch.Tensor,
    logits: torch.Tensor,
):
    loss.backward()
    return logits.grad


def triton_router_backward(
    loss: torch.Tensor,
    logits: torch.Tensor,
):
    loss.backward()
    return logits.grad


def make_forward_factory(
    num_tokens: int, num_keys: int, top_k: int, device: torch.device, dtype: torch.dtype
):
    gen = torch.Generator(device=device).manual_seed(0)
    base = torch.randn(
        2, num_tokens, num_keys, device=device, dtype=dtype, generator=gen
    )

    def _factory(_impl: testing.Implementation):
        logits = base.clone()
        return (logits, num_keys, top_k), {}

    return _factory


def make_backward_factory(
    num_tokens: int, num_keys: int, top_k: int, device: torch.device, dtype: torch.dtype
):
    gen = torch.Generator(device=device).manual_seed(0)
    base = torch.randn(
        2, num_tokens, num_keys, device=device, dtype=dtype, generator=gen
    )

    def _factory(impl: testing.Implementation):
        logits = base.clone().detach().requires_grad_(True)
        if impl.backend == testing.Backend.PYTORCH:
            loss = pytorch_router_forward(logits, num_keys, top_k).sum()
        elif impl.backend == testing.Backend.TRITON:
            loss = triton_router_forward(logits, num_keys, top_k).sum()
        return (loss, logits), {}

    return _factory


@pytest.mark.parametrize(
    "dtype",
    [torch.float32],
)
@pytest.mark.parametrize(
    "case",
    [
        (4096, 64, 8),
        (4096, 128, 16),
        (8192, 64, 8),
        (8192, 128, 16),
        (16384, 64, 8),
        (16384, 128, 16),
        (32768, 64, 8),
        (32768, 128, 16),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_router_forward_throughput(
    dtype: torch.dtype, case: tuple[int, int, int]
) -> None:
    num_tokens, num_keys, top_k = case
    device = torch.device("cuda")

    num_expert = num_keys * num_keys
    print(
        f"[router forward] num_tokens={num_tokens}, "
        f"num_expert={num_expert}, select_expert={top_k}"
    )

    impls = testing.get_impls(
        pytorch_impl=pytorch_router_forward,
        triton_impl=triton_router_forward,
    )
    flops = 4.0 * 2 * num_tokens * num_keys * num_keys
    config = testing.BenchmarkConfig(warmup=5, repeat=1_000)
    results = testing.run_benchmarks(
        impls,
        make_forward_factory(num_tokens, num_keys, top_k, device, dtype),
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
        (4096, 64, 8),
        (8192, 64, 8),
        (16384, 64, 8),
        (32768, 64, 8),
        (4096, 128, 16),
        (8192, 128, 16),
        (16384, 128, 16),
        (32768, 128, 16),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_router_backward_throughput(
    dtype: torch.dtype, case: tuple[int, int, int]
) -> None:
    num_tokens, num_keys, top_k = case
    device = torch.device("cuda")

    num_expert = num_keys * num_keys
    print(
        f"[router backward] num_tokens={num_tokens}, "
        f"num_expert={num_expert}, select_expert={top_k}"
    )

    impls = testing.get_impls(
        pytorch_impl=pytorch_router_backward,
        triton_impl=triton_router_backward,
    )
    flops = 2.0 * 4.0 * 2 * num_tokens * num_keys * num_keys
    config = testing.BenchmarkConfig(warmup=5, repeat=1_000)
    results = testing.run_benchmarks(
        impls,
        make_backward_factory(num_tokens, num_keys, top_k, device, dtype),
        flops=flops,
        config=config,
    )

    testing.show_benchmarks(results)
