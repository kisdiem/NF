# Local Electrical NF

这是一个新增的局部电信号神经场实验版本。它不修改现有 `dynamic_nf.py`，后者仍作为 Dynamic NF 对照。

## 核心数据流

```text
784 -> Linear(64) -> reshape [B,1,8,8]
    -> membrane potential V
    -> differentiable threshold release
    -> fixed local 3x3 electrical propagation
    -> membrane decay + incoming accumulation (4 steps)
    -> flatten [B,64] -> Linear(10)
```

节点状态是 `V`；每个节点还有可训练的阈值 `T`、传播强度 `S`、兴奋/抑制符号和衰减系数。默认使用 sigmoid surrogate：

```text
release_i(t) = sigmoid((V_i(t)-T_i)/tau) * S_i
incoming(t)  = K_local * release(t)
V(t+1)      = decay_i * V(t) + incoming(t)
```

其中 `K_local` 是固定的 3×3 kernel：正交邻居为 1，对角邻居为 0.7，中心为 0。边界使用零填充。传播由 `conv2d` 完成，没有逐节点 Python 循环。

## Tensor shape

| 对象 | shape |
|---|---|
| 输入 | `[B,784]` |
| 投影后场 | `[B,1,8,8]` |
| V / threshold / strength / decay | `[B,1,8,8]` 或可广播的 `[1,1,8,8]` |
| release / incoming | `[B,1,8,8]` |
| 输出读出 | `[B,64] -> [B,10]` |

整个实现不存在 `N×N` relation tensor，也没有 Q/K、QK^T、attention 或 GNN library。每个节点最多接收固定 8 个局部邻居。

## 复杂度

默认 8×8 网格共有 64 个节点、每步 8 个邻居。局部传播卷积的粗略乘加 FLOPs 为：

```text
8*8*3*3*2 = 1,152 FLOPs / step
4 steps    = 4,608 FLOPs / sample
```

这是局部传播部分的估计，不包含输入/输出 Linear 和阈值、衰减等逐元素操作。复杂度随节点数和固定邻居数线性增长，而不是全局关系的平方增长。

## 消融

训练脚本中的模型名：

* `local`：完整 Local Electrical NF
* `no_threshold`：释放门固定为 1
* `no_persistence`：不保留上一时刻膜电位
* `no_inhibition`：只允许正向贡献
* `one_step`：只运行一个内部时间步

命令：

```powershell
py -3 train_local_electrical_nf.py --epochs 10 --subset 5000 --batch 128 --steps 4 --device cpu
```

结果写入 `local_electrical_nf_results/results_seed0.json`，Local NF 的膜电位曲线、状态变化和最后一个测试 batch 的状态热图也会写入同目录。

## 初次结果

统一条件：MNIST，训练子集 5000，测试集 10000，hidden=64，Adam，10 epochs，CPU，seed=0。以下是 test accuracy 的最高值/第 10 轮值；参数量几乎相同。

| 模型 | best | final | 参数量 | CPU 秒 |
|---|---:|---:|---:|---:|
| Linear | 90.61% | 90.43% | 50,890 | 75.58 |
| ReLU | 92.40% | 92.40% | 50,890 | 77.61 |
| GELU | 92.13% | 92.13% | 50,890 | 76.50 |
| Dynamic NF | 92.53% | 92.53% | 51,183 | 127.08 |
| Local Electrical NF | 92.56% | 92.56% | 51,146 | 88.03 |
| no threshold | 90.59% | 90.59% | 51,146 | 88.72 |
| no persistence | 90.92% | 90.92% | 51,146 | 90.17 |
| no inhibition | **92.87%** | **92.87%** | 51,146 | 88.86 |
| one step | 92.11% | 92.11% | 51,146 | 82.25 |

## 客观结论

1. 最基本机制可以训练：Local NF 明显高于 Linear，且没有状态爆炸或 NaN。
2. 在这次 10 轮、5000 样本实验中，Local NF 与 Dynamic NF 持平，略高于 ReLU，但提升很小，尚不能证明局部电信号机制本身优于传统激活。
3. 膜电位持续性和阈值都很重要：去掉任一项后准确率下降约 1.6～2.0 个百分点。
4. 单步低于多步，说明内部演化至少提供了可测的增益。
5. no-inhibition 反而最高，说明当前版本的抑制符号设计还没有形成有效的竞争机制，不能把它解释成“抑制有贡献”。这应作为下一轮重点，而不是隐藏该结果。
6. 目前每个节点的 `T/S/decay/sign` 是独立可训练参数，但局部 kernel 本身固定；因此当前验证的是“可学习膜电位动力学 + 固定局部电传播”，不是完整可学习突触拓扑。

## no-inhibition 延长训练

在相同配置下仅将 `no_inhibition` 延长到 20 epochs：

```text
subset=5000, batch=128, steps=4, seed=0, CPU
```

测试集准确率在第 13 轮达到 93.12%，第 16 轮达到最高 **93.47%**，第 20 轮为 **93.44%**。训练集在第 18 轮后达到 100%，但测试集仍在 93.4% 附近缓慢上升，说明此时已经接近过拟合边缘，但还没有出现明显的测试性能崩落。

这次延长训练提高了上一轮 10 epochs 的 92.87% 最高值约 0.60 个百分点；不过它仍然不能单独证明局部电场优于 ReLU，因为此前 ReLU 只做过 10 轮，后续应在同样 20 轮条件下补跑公平对照。

## 统一 25 轮对照

为避免训练轮数造成偏差，重新从相同 seed、相同初始化和相同数据顺序开始，对主要模型统一训练 25 轮：

| 模型 | 最高 test | 第25轮 test |
|---|---:|---:|
| Linear | 90.61% | 89.24% |
| ReLU | 93.27% | 93.17% |
| GELU | 93.19% | 93.06% |
| Dynamic NF | 93.14% | 92.97% |
| Local Electrical NF | 92.99% | 92.99% |
| no-inhibition Local NF | **93.47%** | 93.32% |

因此在这组固定 seed 的 25 轮实验中，no-inhibition Local NF 仍是最高，但它只比 ReLU 高 0.20 个百分点；完整 Local NF 反而低于 ReLU、GELU 和 Dynamic NF。测试曲线在训练集接近 100% 后基本进入平台，继续训练没有带来稳定提升。
