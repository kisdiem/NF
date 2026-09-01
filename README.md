# BioNeuron / BioNF

本分支是 BioNeuron 的干净研究分支，只保留单神经元内部多树突计算相关的模型、核心实验与关键结果。Local/Dynamic/Hierarchical/Temporal NF 等历史探索请看 `main` 或其他分支。

## 核心问题

> 传统人工神经元只有一次加权求和和一个点式激活。如果一个神经元内部具有多个树突分支、分支级非线性和胞体整合，它能否在参数匹配条件下形成更高的表达效率或泛化能力？

当前结构：

```text
input projection
 -> multiple dendritic branches
 -> branch-wise nonlinear transform
 -> excitation / inhibition
 -> soma membrane potential
 -> adaptive threshold
 -> activation
 -> classifier
```

详细设计与阶段结论见 [`docs/BIO_DESIGN_AND_RESULTS.md`](docs/BIO_DESIGN_AND_RESULTS.md)。

## 当前证据

| Task | Linear | ReLU | GELU | BioNeuron |
|---|---:|---:|---:|---:|
| XOR | 50.0% | 100.0% | 100.0% | 100.0% |
| Circles | 49.7% | 100.0% | 100.0% | 100.0% |
| Two Moons | 92.8% | 99.9% | 98.3% | 100.0% |

MNIST subset=5000，20 epochs：seed 0 Bio best 约 93.3%，seed 1 约 93.7%。但当前 Bio 参数量约为对应 ReLU MLP 的 1.66 倍，因此不能据此宣称 Bio 更强。

XOR 消融：4 branches=100%，1 branch=72.3%，one-step=75%；no-temporal/no-inhibition/fixed-threshold 仍可到 100%。当前最值得研究的核心因此收缩为：

> **多树突分支 + 分支内部非线性 + 胞体整合。**

## 目录

```text
models/bio_neuron.py
experiments/bio_experiments.py
docs/BIO_DESIGN_AND_RESULTS.md
results/
```

`results/` 只保留 XOR/Circles/Moons 和 MNIST 关键 seed JSON，不保留大量中间图、重复优化结果和无关模型结果。

## 运行

```bash
pip install -r requirements.txt
python experiments/bio_experiments.py --task xor --epochs 300
python experiments/bio_experiments.py --task mnist --epochs 20 --subset 5000
```

树突数量：

```bash
python experiments/bio_experiments.py --task xor --branch-sweep 1,2,4,8 --epochs 300
```

参数量匹配 baseline：

```bash
python experiments/bio_experiments.py --task mnist --parameter-match --epochs 20 --subset 5000
```

核心消融：

```bash
python experiments/bio_experiments.py --task xor --ablation --epochs 300
```

## 下一阶段

只优先验证三件事：

1. parameter-matched Bio vs ReLU/GELU，至少 5 seeds；
2. branches=1/2/4/8，在总参数受控条件下验证分支结构本身；
3. 在比 XOR/MNIST 更能体现组合结构的任务上验证，例如 Checkerboard、Parity16/20、带噪组合任务。

若优势在参数匹配和多 seed 下消失，就不再继续为 Bio 叠加额外生物机制。
