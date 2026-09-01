# BioNeuron / BioNF 专题分支范围

本分支只把 **BioNeuron / BioNF** 作为当前主线。仓库中仍存在从 `main` 继承的 Local NF、Dynamic NF、Temporal NF 等历史文件，它们仅为历史兼容，不属于本分支当前研究结论。

## 当前入口

1. `README.md`：分支首页、当前结论、复现命令。
2. `docs/BIO_DESIGN_AND_RESULTS.md`：从 `main` 迁移并整合的 BioNeuron 设计、消融、失败诊断和阶段结果。
3. `README_BioNeuron.md`：第一阶段原始实验说明。
4. `models/bio_neuron.py`：BioNeuron 核心实现。
5. `experiments/bio_experiments.py`：XOR / Circles / Moons / MNIST 实验入口。

## 当前主线结构

```text
input projection
 -> multiple dendritic branches
 -> branch-wise nonlinear transform
 -> excitation / inhibition integration
 -> soma membrane potential
 -> adaptive threshold / adaptation
 -> activation
 -> classifier
```

## 代表结果

- XOR / Circles / Moons：BioNeuron 均能形成非线性表示，但 ReLU/GELU 也能解决这些简单任务。
- MNIST subset=5000，20 epochs：BioNeuron seed 0 best 约 93.3%，seed 1 best 约 93.7%。
- MNIST 30 epochs：seed 0 约 93.39%，seed 1 约 93.66%。
- XOR 消融中，4 branches -> 1 branch 后从 100% 降到 72.3%；one-step 为 75%。
- no temporal / no inhibition / fixed threshold 在 XOR 上仍为 100%，所以这些机制尚不能作为独立性能贡献。

原始结果集中在 `bio_results/`。

## 当前判断

当前最值得继续验证的核心是：

> **多树突分支 + 分支级非线性 + 胞体整合。**

BioNeuron 目前还不能宣称优于 ReLU/GELU，因为参数量约为对应 ReLU MLP 的 1.66 倍，而且完整消融主要集中在简单 XOR。

下一阶段优先做 parameter-matched baseline、1/2/4/8 branch 曲线、困难任务、困难任务消融、参数效率和样本效率。
