# Dynamic Neural Field 实验

该版本在现有项目旁新增 `DynamicNeuralField`，不修改原始 NF。它使用固定容量的隐藏状态：

```text
Linear(784,64) -> reshape [B,16,4]
-> G_t [B,16,16] = tanh(Q(H_t)K(H_t)^T / temperature) * gain
-> message = G_t^T @ H_t
-> 多分支局部 tanh 更新
-> gated state persistence + LayerNorm
-> flatten [B,64] -> Linear(64,10)
```

关系矩阵是有符号、非归一化、样本相关的有向矩阵；所有节点使用同步更新。默认最多运行 4 步，允许反馈。实现位于 [`dynamic_nf.py`](dynamic_nf.py)，训练入口位于 [`train_dynamic_nf.py`](train_dynamic_nf.py)。

## 运行

```powershell
py -3 train_dynamic_nf.py --epochs 5 --subset 5000 --batch 128
py -3 train_dynamic_nf.py --epochs 20 --subset 5000 --batch 128
```

输出写入 `dynamic_nf_results/`，包括每个消融版本的逐轮 JSON、关系矩阵热图和 state-change 曲线。

## 消融开关

入口默认依次运行：

- Linear / ReLU / GELU
- Dynamic NF
- fixed relation
- one-step
- no feedback
- no state
- no branches
- parameter-matched MLP

## 已完成结果

MNIST 子集 5000、20 轮、hidden=64、batch=128、seed=0：

| 版本 | 最高准确率 | 参数量 |
|---|---:|---:|
| Linear | 90.61% | 50,890 |
| ReLU | 93.16% | 50,890 |
| GELU | 93.06% | 50,890 |
| Dynamic NF | 93.14% | 51,183 |
| Fixed relation | 93.71% | 51,183 |
| One-step | 93.77% | 51,183 |
| No feedback | 93.12% | 51,183 |
| No state | 91.71% | 51,183 |
| No branches | 92.85% | 51,048 |
| Parameter-matched MLP | 93.16% | 50,890 |

当前结果没有证明多步动态关系优于固定关系或单步更新；状态保留是最明确有贡献的机制。失败结果和中间诊断均保留。

## 全量 MNIST 对照

完整训练集 60,000、完整测试集 10,000、hidden=64、batch=128、10 轮、seed=0：

| 版本 | 最高准确率 | 最终准确率 |
|---|---:|---:|
| Linear | 92.29% | 91.88% |
| ReLU | 97.60% | 97.30% |
| GELU | **97.65%** | **97.65%** |
| Dynamic NF | 97.36% | 97.13% |

全量数据下 Dynamic NF 明显超过 Linear，但仍低于 ReLU/GELU；因此目前它的动态关系机制尚未带来超过普通 MLP 的收益。
