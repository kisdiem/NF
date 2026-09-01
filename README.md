# NF / BioNeuron Branch

本分支专门维护 **BioNeuron / BioNF**。空间 Local NF 研究请切换到 `local` 分支；完整历史与其他探索版本保留在 `main`。

## 当前研究目标

BioNeuron 研究的核心问题是：

> 如果一个人工神经元内部包含多个树突分支、分支级非线性、兴奋/抑制整合、膜电位和阈值适应，它是否能形成传统点神经元没有的计算能力，并在参数匹配条件下表现出更好的表达效率或泛化能力？

当前结构：

```text
输入投影
  -> 多树突分支局部非线性
  -> 兴奋 / 抑制整合
  -> 胞体膜电位
  -> 动态阈值 / adaptation
  -> activation
  -> 分类器
```

完整设计、失败诊断和实验结论：[`docs/BIO_DESIGN_AND_RESULTS.md`](docs/BIO_DESIGN_AND_RESULTS.md)。

## 当前主要结果

### 合成任务

| 任务 | Linear | ReLU | GELU | BioNeuron |
|---|---:|---:|---:|---:|
| XOR | 50.0% | 100.0% | 100.0% | **100.0%** |
| Circles | 49.7% | 100.0% | 100.0% | **100.0%** |
| Two Moons | 92.8% | 99.9% | 98.3% | **100.0%** |

这些任务只说明 BioNeuron 能形成非线性表示，不能证明它优于 ReLU/GELU。

### MNIST subset=5000

| 配置 | Linear | ReLU | GELU | BioNeuron |
|---|---:|---:|---:|---:|
| 10 epochs | 90.4% | 92.4% | 92.3% | 92.6% |
| 20 epochs, seed 0 best | 90.4% | 92.9% | 93.2% | **93.3%** |
| 20 epochs, seed 1 best | 90.5% | 93.2% | 93.1% | **93.7%** |

30 轮记录：

- BioNeuron seed 0：约 93.39%
- BioNeuron seed 1：约 93.66%

当前 BioNeuron 参数量约为对应 ReLU MLP 的 1.66 倍，因此这些差异不能作为“Bio 更强”的证据。

## 关键消融

XOR：

| 变体 | 测试准确率 |
|---|---:|
| Full BioNeuron | 100.0% |
| 1 branch | 72.3% |
| no temporal | 100.0% |
| no inhibition | 100.0% |
| fixed threshold | 100.0% |
| one step | 75.0% |

目前最值得继续验证的是：

> **多树突分支 + 分支内部非线性 + 胞体整合。**

抑制、动态阈值和时间机制在简单 XOR 上还没有显示独立必要性。

## 一次重要失败

第一版 MNIST 曾出现：

```text
soma_v_mean            ~= -5.18
activation_rate        ~= 0.00002
dead_neuron_ratio      = 1.0
saturated_neuron_ratio = 1.0
```

模型几乎所有神经元进入负饱和。通过重新平衡兴奋/抑制初始化，并把多树突贡献从直接求和改为平均，BioNeuron 从约 11.4% 恢复到 92.6%。

这说明内部动力学确实影响训练，但也说明 BioNeuron 对尺度设计比普通激活函数更敏感。

## 核心文件

```text
BIO_BRANCH_SCOPE.md
README_BioNeuron.md
docs/BIO_DESIGN_AND_RESULTS.md

models/bio_neuron.py
experiments/bio_experiments.py
requirements_bio.txt

bio_results/xor_results.json
bio_results/circles_results.json
bio_results/moons_results.json
bio_results/mnist_results_seed0.json
bio_results/mnist_results_seed1.json
bio_results/mnist_results_seed0_30.json
bio_results/mnist_results_seed1_30.json
```

## 复现实验

XOR：

```bash
python experiments/bio_experiments.py --task xor --epochs 300
```

XOR 消融：

```bash
python experiments/bio_experiments.py --task xor --epochs 300 --ablation
```

MNIST：

```bash
python experiments/bio_experiments.py --task mnist --epochs 20 --subset 5000 --seed 0
python experiments/bio_experiments.py --task mnist --epochs 20 --subset 5000 --seed 1
```

树突非线性对照：

```bash
python experiments/bio_experiments.py --task xor --dendrite soft_threshold
python experiments/bio_experiments.py --task xor --dendrite quadratic
python experiments/bio_experiments.py --task xor --dendrite tanh
```

## 当前优先级

下一阶段最重要的不是继续增加生物机制，而是验证核心结构是否真的有效：

1. parameter-matched ReLU / GELU；
2. branches = 1 / 2 / 4 / 8，并做固定总参数量版本；
3. Checkerboard、Noisy Spiral100、Noisy Moons100；
4. Parity16 / Parity20，避免 Parity8 的组合重复问题；
5. 在困难任务上重新做 branch / temporal / inhibition / threshold 消融；
6. 比较参数效率、样本效率和计算成本。

当前最合理的研究定位是：BioNeuron 已经形成一个值得验证的“多树突人工神经元”假设，但还没有证明它是更优的通用神经网络单元。
