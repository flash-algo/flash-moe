# FlashMoE

**English** | [简体中文](./README_zh.md)

FlashMoE is a high-performance Mixture-of-Experts implementation focused on **fine-grained experts**.


## Why FlashMoE

As LLMs scale, dense Transformer blocks become increasingly expensive. MoE improves parameter efficiency by activating only a subset of parameters per token.
However, existing MoE designs often face a trade-off:

- **Coarse-grained experts** are easier to utilize efficiently, but the number of experts is limited, and under a fixed compute budget they may waste compute or lose information.
- **Fine-grained experts** can scale the number of experts dramatically, but are often bottlenecked by routing quality and memory bandwidth, making stable end-to-end gains hard.


## Key Features

- Router kernel: per-token top-k selection over a Cartesian-product expert space
- Expert kernel: fused gather, activation, and weighted accumulation
- MLP kernel: for the shared dense branch


## Installation

### Requirements

- Python >= 3.8
- PyTorch with CUDA support
- Triton
- transformers

### Install from source

```bash
git clone https://github.com/flash-algo/flash-moe.git
cd flash-moe
pip install -e .
```


## Quick Start

### FlashMoE module

```python
import torch

from flash_moe.modules.flash_moe import FlashMoE, FlashMoEConfig

device = torch.device("cuda")
dtype = torch.bfloat16

config = FlashMoEConfig(
	hidden_size=1024,
	intermediate_size=4096,
	hidden_act="silu",
	num_experts=16384,
	num_experts_per_tok=64,
	norm_topk_prob=False,
)

x = torch.randn(1, 4096, config.hidden_size, device=device, dtype=dtype)
moe = FlashMoE(config).to(device=device, dtype=dtype)

y, router_logits = moe(x)
print(y.shape, router_logits.shape)
```


## Benchmarks

Includes pytest-based kernel benchmarks and tests.

- Router: [docs/flash_router.md](docs/flash_router.md), tests in `tests/test_router.py`
- Expert: [docs/flash_expert.md](docs/flash_expert.md), tests in `tests/test_expert.py`

Run all tests:

```bash
pytest -q
```

Run a specific kernel benchmark:

```bash
pytest tests/test_router.py -s
pytest tests/test_expert.py -s
```


## License

See [LICENSE](LICENSE).
