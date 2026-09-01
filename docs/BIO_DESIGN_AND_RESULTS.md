# BioNeuron：设计与核心实验结果

本文件把 `main` 中与 BioNeuron / BioNF 直接相关的设计、消融、失败诊断和阶段结果集中到 `bio` 分支。原始结果仍保存在 `bio_results/`，这里作为当前研究入口。

## 1. 研究问题

BioNeuron 的核心问题不是空间神经场，而是：

> 传统人工神经元只有一次加权求和和一个点式激活函数；如果一个神经元内部包含多个可独立计算的树突分支、局部非线性和状态整合，是否能形成更强或更高效的计算单元？

因此 Bio 分支与 Local 分支应明确分开：

- Local：研究节点之间的局部传播、空间拓扑和场动力学。
- Bio：研究单个神经元内部的多分支计算、兴奋/抑制、膜电位和阈值适应。

---

## 2. 当前 BioNeuron 结构

数据流：

```text
input
  -> input projection
  -> multiple dendritic branches
  -> branch-wise nonlinear transform
  -> excitation / inhibition integration
  -> soma membrane potential
  -> adaptive threshold / adaptation
  -> activation
  -> classifier
```

与普通点神经元的区别可以写成：

普通神经元：

```text
y = phi(w^T x)
```

BioNeuron 更接近：

```text
y = phi( sum_r g_r(w_r^T x) )
```

其中 `r` 表示树突分支。也就是说，一个神经元内部先进行多路非线性处理，再由胞体整合。

核心实现：`models/bio_neuron.py`。
实验入口：`experiments/bio_experiments.py`。

---

## 3. 主要机制

### 3.1 多树突分支

默认一个 BioNeuron 包含多个独立 branch，每个 branch 有自己的输入权重。它们不是简单复制输入，而是在胞体汇总之前先形成不同的局部响应。

这是当前最值得继续追踪的机制，因为已有消融显示：在 XOR 上，从 4 branches 降到 1 branch 后，准确率从 100% 降到 72.3%。

但这个结果仍有参数量混杂：更多 branch 同时意味着更多参数，因此后续必须做参数匹配。

### 3.2 树突局部非线性

当前支持：

- `soft_threshold`
- `quadratic`
- `tanh`

目的不是单纯更换激活函数，而是让每个分支先形成独立非线性，再进入胞体整合。

### 3.3 兴奋 / 抑制

BioNeuron 将兴奋和抑制贡献分开建模，再形成净输入，而不是只使用一组普通正负权重。

但当前 XOR 消融中去掉 inhibition 仍为 100%，所以暂时不能把“兴奋/抑制”宣称为性能来源。

### 3.4 膜电位与时间展开

当前默认运行多个内部 step，并保留输入 trace、树突状态和胞体膜电位的短期累积。

XOR 上：

- full：100%
- no temporal：100%
- one-step：75%

这说明“完全没有时间机制”与“只运行一步”不是同一件事；当前简单任务只能说明多步展开可能帮助优化或表示，但还不能证明长期状态本身是关键能力。

### 3.5 动态阈值

阈值可以随近期活动发生适应，用于模拟激活后的短时可激发性变化。

但 XOR 上固定阈值仍为 100%，所以该机制目前属于生物合理性设计，而不是已被证明的性能贡献。

---

## 4. 第一阶段结果

### 4.1 合成任务

| 任务 | Linear | ReLU | GELU | BioNeuron |
|---|---:|---:|---:|---:|
| XOR | 50.0% | 100.0% | 100.0% | **100.0%** |
| 同心圆 | 49.7% | 100.0% | 100.0% | **100.0%** |
| Two Moons | 92.8% | 99.9% | 98.3% | **100.0%** |

这些实验只证明 BioNeuron 具备非线性表示能力，不能证明它优于 ReLU/GELU，因为简单任务已经被普通 MLP 解决。

### 4.2 MNIST subset=5000

| 配置 | Linear | ReLU | GELU | BioNeuron |
|---|---:|---:|---:|---:|
| 10 epochs | 90.4% | 92.4% | 92.3% | 92.6% |
| 20 epochs, seed 0 best | 90.4% | 92.9% | 93.2% | **93.3%** |
| 20 epochs, seed 1 best | 90.5% | 93.2% | 93.1% | **93.7%** |

30 轮记录中，BioNeuron：

- seed 0：约 93.39%
- seed 1：约 93.66%

这些差异仍不足以宣称 BioNeuron 更强，因为当前 Bio 参数量约为对应 ReLU MLP 的 1.66 倍。

原始结果位于：

- `bio_results/mnist_results_seed0.json`
- `bio_results/mnist_results_seed1.json`
- `bio_results/mnist_results_seed0_30.json`
- `bio_results/mnist_results_seed1_30.json`

---

## 5. 一次重要失败与修正

第一版 MNIST 中出现：

```text
soma_v_mean            ~= -5.18
activation_rate        ~= 0.00002
dead_neuron_ratio      = 1.0
saturated_neuron_ratio = 1.0
```

模型几乎所有神经元都进入负饱和，测试准确率一度接近随机。

随后进行了两项关键修正：

1. 重新平衡 excitation / inhibition 初始化。
2. 胞体对多个树突分支由直接求和改为平均，避免 branch 数量直接放大膜电位尺度。

修正后 BioNeuron 从约 11.4% 恢复到 92.6%。

这说明 BioNeuron 的内部动力学确实会影响训练，但也说明其尺度设计比普通激活函数更敏感。

另外，早期曾出现约 81% 的错误对照，原因是把 5000 个训练样本作为单一 batch，每个 epoch 只更新一次；修正为 mini-batch 后，Linear 恢复到约 90.4%。该错误结果不应再用于比较。

---

## 6. XOR 消融

| 变体 | 测试准确率 |
|---|---:|
| Full BioNeuron | 100.0% |
| 1 branch | 72.3% |
| no temporal | 100.0% |
| no inhibition | 100.0% |
| fixed threshold | 100.0% |
| one step | 75.0% |

当前最合理的解释：

- **树突分支是最明显的候选有效机制。**
- 时间展开可能有作用，但简单 XOR 还无法区分“真正的时序能力”和“多次内部非线性计算”。
- inhibition、adaptive threshold 在 XOR 上没有显示必要性。

因此后续研究应避免把所有生物机制一起作为贡献，而应把重点收缩到“多树突分支 + 分支内部非线性 + 胞体整合”。

---

## 7. 当前最重要的方法学限制

### 7.1 参数量不公平

这是当前最大限制。

如果 BioNeuron 用 4 branches，而 ReLU 只有单个 hidden layer，那么 Bio 获得更多参数是天然的。现有 93.3%-93.7% 对 92.9%-93.2% 的结果还不能证明结构本身更好。

必须补：

```text
BioNeuron
vs parameter-matched ReLU
vs parameter-matched GELU
vs width-matched 1-branch Bio
```

### 7.2 简单任务饱和

XOR、Circles、Moons 已经被普通 MLP 接近完全解决，继续在这些任务上加机制没有太大辨别力。

后续需要更困难的组合非线性、高噪声和泛化任务。

### 7.3 消融目前只在 XOR 上最完整

“no temporal / no inhibition / fixed threshold 没影响”只能解释 XOR，不能外推到 MNIST 或其他任务。

---

## 8. 后续决定性实验

建议按以下顺序：

1. **参数匹配**：Bio vs ReLU/GELU，保持总参数量近似相同。
2. **branch 数量曲线**：1 / 2 / 4 / 8 branches，并同步做参数固定版本。
3. **困难任务**：Checkerboard、Noisy Spiral100、Noisy Moons100，以及重新设计的 Parity16/Parity20。
4. **困难任务消融**：branch、temporal、inhibition、adaptive threshold 分别去掉。
5. **样本效率**：在较少训练样本下比较 Bio 与普通 MLP，验证是否存在结构归纳偏置优势。
6. **参数效率**：给定固定参数预算，比较可达到的准确率。
7. **计算成本**：记录训练/推理时间和 FLOPs，不只比较准确率。

---

## 9. 当前研究定位

现阶段最值得保留的核心假设是：

> **一个人工神经元内部包含多个独立非线性树突子单元，可能比传统点神经元具有不同的参数利用方式和组合表达能力。**

尚未被证明的是：

- 是否在参数匹配条件下稳定优于 ReLU/GELU；
- 是否能在真实复杂任务上获得优势；
- 时间、抑制、动态阈值是否提供独立贡献。

因此目前 BioNeuron 是“值得继续验证的机制假设”，而不是已经证明的新型通用神经网络单元。
