# Flash Expert

The **flash expert** is an accelerated primitive for Mixture-of-Experts expert evaluation when routing has already been computed.
Given per-token expert indices and routing weights, it gathers the corresponding expert parameters and computes a fused “expert MLP-like” mapping:

- compute per-token and per-expert scores via a dot product with a **down** weight,
- apply a SiLU nonlinearity and multiply by the routing weight,
- project back to hidden space via an **up** weight and sum over the selected experts.

This is implemented as Triton kernels with an autograd-aware wrapper.


## Kernel Interface

The primary user-facing API is the autograd-aware wrapper:

```python
from flash_moe.ops.flash_expert import triton_flash_expert_func

expert_states = triton_flash_expert_func(
    hidden_states,
    down_weights,
    up_weights,
    indices,
    routing_weights,
)
```

## Arguments

- `hidden_states` (`torch.Tensor`):
	- shape: `(num_tokens, hidden_size)`
	- dtype: typically `torch.float16`, `torch.bfloat16`, or `torch.float32`
    - device: CUDA tensor (Triton kernels run on GPU)
- `down_weights` (`torch.Tensor`):
	- shape: `(num_experts, hidden_size)`
	- dtype: typically `torch.float16`, `torch.bfloat16`, or `torch.float32`
    - device: CUDA tensor (Triton kernels run on GPU)
- `up_weights` (`torch.Tensor`):
	- shape: `(num_experts, hidden_size)`
	- dtype: typically `torch.float16`, `torch.bfloat16`, or `torch.float32`
    - device: CUDA tensor (Triton kernels run on GPU)
- `indices` (`torch.LongTensor`):
	- shape: `(num_tokens, num_experts_per_tok)`
	- note: each entry is an expert id in `[0, num_experts)`
    - device: CUDA tensor (Triton kernels run on GPU)
- `routing_weights` (`torch.Tensor`):
	- shape: `(num_tokens, num_experts_per_tok)`
	- dtype: typically `torch.float16`, `torch.bfloat16`, or `torch.float32`
    - device: CUDA tensor (Triton kernels run on GPU)


## Returns

- `expert_states` (`torch.Tensor`):
	- shape: `(num_tokens, hidden_size)`
	- dtype: matches `hidden_states.dtype`
	- device: same as inputs

## Testing

Expert tests and benchmarks live in [tests/test_expert.py](../tests/test_expert.py).
They include a PyTorch reference implementation and Triton-based implementations for forward and backward throughput.

To run the expert benchmarks on a CUDA-enabled machine:

```bash
pytest tests/test_expert.py -s
```

You can run individual tests with, for example:

```bash
pytest tests/test_expert.py::test_expert_forward_backward_throughput -s
pytest tests/test_expert.py::test_expert_backward_throughput -s
```

Make sure that:

- PyTorch is installed with CUDA support,
- Triton is installed and compatible with your CUDA/PyTorch version,
- the GPU has sufficient memory for the chosen `(num_tokens, hidden_size, num_experts, top_k)` settings.
