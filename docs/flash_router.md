# Flash Router

The **flash router** is a accelerated primitive for computing the top‑k pairwise routing scores between two sets of logits. It is designed for high‑throughput expert routing or key–value selection in Mixture‑of‑Experts style architectures.

For each token, the router takes two 1D logit vectors of length $N$ and forms all $N^2$ pairwise combinations, then returns the top‑k scores and their indices. The computation is fused in a single GPU kernel that:

- materializes the pairwise scores implicitly inside the Triton kernel,
- finds the top‑k pairs per token,
- writes out only the top‑k scores and flattened pair indices.


## Kernel Interface

The primary user‑facing API is the autograd‑aware wrapper:

```python
from flash_moe.ops.flash_router import triton_flash_router_func

scores, indices = triton_flash_router_func(router_logits, num_keys, top_k)
```

**Arguments**

- `router_logits` (`torch.Tensor`):
	- shape: `(2, num_tokens, num_keys)`;
	- dtype: typically `torch.float32` (internally cast to `float32`);
	- device: CUDA tensor (the Triton kernels run on GPU).
- `num_keys` (`int`):
	- number of keys $N$ per token; must equal `router_logits.size(-1)`.
- `top_k` (`int`):
	- number of pairwise combinations to select per token; must satisfy `0 <= top_k <= num_keys`.

**Returns**

- `scores` (`torch.Tensor`):
	- shape: `(num_tokens, top_k)`;
	- dtype: same as input `router_logits` (the kernel runs in `float32` and casts back).
- `indices` (`torch.LongTensor`):
	- shape: `(num_tokens, top_k)`;
	- flattened pair indices $k = i \cdot N + j$ with `0 <= k < num_keys * num_keys`.


## Testing

Router tests and benchmarks live in `tests/test_router.py`. They provide both a pure‑Python reference implementation and Triton‑based implementations to compare against.

To run the router benchmarks on a CUDA‑enabled machine:

```bash
pytest tests/test_router.py -s
```

You can run individual tests with, for example:

```bash
pytest tests/test_router.py::test_router_forward_throughput -s
pytest tests/test_router.py::test_router_backward_throughput -s
```

Make sure that:

- PyTorch is installed with CUDA support,
- Triton is installed and compatible with your CUDA/PyTorch version,
- the environment has a GPU and sufficient memory for the chosen `(num_tokens, num_keys, top_k)` settings.
