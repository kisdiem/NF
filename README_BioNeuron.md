# Bio-Neuron / BioNF 第一阶段实验

这是一个独立于原 Neural Field 的最小生物神经元启发型模块。它不使用空间邻接传播，而是把一个神经元拆成：

```text
输入投影 -> 多个树突分支局部非线性 -> 兴奋/抑制 -> 膜电位 -> 动态阈值 -> activation
```

核心实现位于 [`models/bio_neuron.py`](models/bio_neuron.py)，实验入口位于 [`experiments/bio_experiments.py`](experiments/bio_experiments.py)。

## 一键运行

```powershell
py -3 experiments\bio_experiments.py --task xor --epochs 300
py -3 experiments\bio_experiments.py --task circles --epochs 200
py -3 experiments\bio_experiments.py --task moons --epochs 200
py -3 experiments\bio_experiments.py --task mnist --epochs 10 --subset 5000
```

运行全部任务：

```powershell
py -3 experiments\bio_experiments.py --task all --epochs 200
```

XOR 消融：

```powershell
py -3 experiments\bio_experiments.py --task xor --epochs 300 --ablation
```

树突非线性可选：

```powershell
py -3 experiments\bio_experiments.py --task xor --dendrite soft_threshold
py -3 experiments\bio_experiments.py --task xor --dendrite quadratic
py -3 experiments\bio_experiments.py --task xor --dendrite tanh
```

硬 spike 对照：

```powershell
py -3 experiments\bio_experiments.py --task xor --hard-spike
```

## 当前已运行结果

配置：CPU、隐藏维度 16（合成任务）或 64（MNIST）、Adam、相同训练轮数和学习率。

| 任务 | Linear | ReLU | GELU | BioNeuron |
|---|---:|---:|---:|---:|
| XOR | 50.0% | 100.0% | 100.0% | **100.0%** |
| 同心圆 | 49.7% | 100.0% | 100.0% | **100.0%** |
| Two Moons | 92.8% | 99.9% | 98.3% | **100.0%** |
| MNIST 子集 5000，10 轮 | 90.4% | 92.4% | 92.3% | 92.6% |
| MNIST 子集 5000，20 轮，seed 0（最高） | 90.4% | 92.9% | 93.2% | **93.3%** |
| MNIST 子集 5000，20 轮，seed 1（最高） | 90.5% | 93.2% | 93.1% | **93.7%** |

合成任务说明 BioNeuron 确实能产生线性模型没有的表示：它稳定解决了 XOR 和同心圆。MNIST 第一版则失败，诊断显示：

```text
soma_v_mean             ~= -5.18
activation_rate         ~= 0.00002
dead_neuron_ratio       = 1.0
saturated_neuron_ratio  = 1.0
```

MNIST 对照使用 mini-batch=128，每轮约 40 次更新；此前曾出现 81% 的错误结果，是因为比较脚本把整个子集作为一个 batch，每轮只更新一次。修正训练循环后，Linear 恢复到 90.4%，与原实验一致。随后修正兴奋/抑制初始化并将胞体分支贡献改为平均，BioNeuron 从 11.4% 提升到 92.6%。

## XOR 消融结果

| 消融 | 测试准确率 |
|---|---:|
| 完整模型 | 100.0% |
| 去掉树突分支（1 branch） | 72.3% |
| 去掉时间累积 | 100.0% |
| 去掉抑制 | 100.0% |
| 固定阈值 | 100.0% |
| 只运行 1 个时间步 | 75.0% |

初步结论：树突分支是最关键的机制；静态树突组合已经能够形成非线性，动态阈值和抑制在这个简单 XOR 上尚未显示必要性；少量时间展开有帮助，但并不是唯一来源。

## 输出文件

所有结果默认写入 `bio_results/`：

- `*_results.json`：loss、accuracy、参数量、耗时、FLOPs 估算和 BioNeuron 诊断
- `*_hidden.png`：隐藏空间 PCA 投影
- `*_dendritic_weights.png`：训练后的树突有效权重热图

FLOPs 是结构级近似值，不是硬件 profiler 的精确计数；时间为当前机器上的端到端训练时间。

## 当前客观结论

1. BioNeuron 已通过 Level 1 的基本非线性能力检查：XOR 接近 100%。
2. 它在合成任务上明显超过纯线性模型。
3. 目前还没有达到或超过 ReLU 的证据。
4. 第一版 MNIST 失败，原因是状态累积导致胞体整体进入负饱和区。
5. 修正尺度后 10 轮 BioNeuron 达到 92.6%；20 轮 seed 0/1 的最高准确率分别为 93.3%/93.7%，略高于对应 ReLU，但参数量约为其 1.66 倍，不能宣称已经在参数效率上胜出。
6. 当前需要继续做参数量匹配和消融，确认提升来自树突机制，而不是额外参数。
