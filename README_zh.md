# FlashMoE

[English](./README.md) | **简体中文**

FlashMoE 是一个高性能的 Mixture-of-Experts 实现, 重点关注 **细粒度专家**.


## 为什么 FlashMoE

随着大语言模型规模增长, 稠密 Transformer block 的成本会快速上升. MoE 通过每个 token 仅激活一部分参数提升参数效率.
但现有 MoE 设计常常面临取舍:

- **粗粒度专家** 更容易高效利用算力, 但专家数量受限, 并且固定计算预算下可能会造成算力浪费或丢失信息.
- **细粒度专家** 可以把专家数量扩展到非常大, 但往往受路由质量与显存带宽瓶颈影响, 难以获得稳定的端到端收益.

FlashMoE 试图在两者之间取得更好的平衡:

- 将 **细粒度专家** 表达为紧凑的权重表.
- 使用 **笛卡尔积路由** 高效搜索超大专家空间.
- 采用 **专家中心调度** 改善访存局部性与吞吐.
- 采用 **混合设计**: 共享的稠密 MLP 与路由得到的细粒度专家并行叠加.


## 主要特性

- 路由内核: 在笛卡尔积专家空间上做 per-token top-k 选择
- 专家内核: 融合 gather, 激活, 加权累加
- MLP 内核: 用于共享稠密分支


## 安装

### 环境要求

- Python >= 3.8
- 支持 CUDA 的 PyTorch
- Triton
- transformers

### 从源码安装

```bash
git clone https://github.com/flash-algo/flash-moe.git
cd flash-moe
pip install -e .
```


## 快速开始

### FlashMoE 模块

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


## 基准测试

包含基于 pytest 的 kernel 基准与测试.

- Router: 说明见 [docs/flash_router.md](docs/flash_router.md), 测试在 `tests/test_router.py`
- Expert: 说明见 [docs/flash_expert.md](docs/flash_expert.md), 测试在 `tests/test_expert.py`

运行所有测试:

```bash
pytest -q
```

运行单个 kernel benchmark:

```bash
pytest tests/test_router.py -s
pytest tests/test_expert.py -s
```


## License

见 [LICENSE](LICENSE).
