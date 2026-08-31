# Temporal Dynamic NF 实验

本实验把当前 Dynamic NF 作为 recurrent cell 使用：每个外部时间步注入新输入，维护持久状态 `[B,16,4]`，然后执行一次同步 Dynamic NF 更新。没有引入 Transformer 或 GNN library。

任务：

- `seq`：MNIST 按行输入，28 个时间步，每步 28 个像素；
- `perm`：固定随机置换后的按行输入；
- `moving`：由 MNIST 数字生成 10 帧确定性平移序列。

运行：

```powershell
py -3 train_temporal_nf.py --task all --epochs 5 --subset 5000 --batch 128
py -3 train_temporal_nf.py --task seq --pixelwise --epochs 5 --subset 5000
```

## 修正后的结果

MNIST 子集 5000、batch=128、5 轮、seed=0。Linear/ReLU 使用统一的 `state = 0.8*old + 0.2*input` 递推；GRU 和 Dynamic NF 使用同样的数据顺序和训练条件。

| 任务 | Linear | ReLU | GRU | Dynamic NF |
|---|---:|---:|---:|---:|
| Sequential MNIST | 35.95% | 49.54% | **80.53%** | 48.50% |
| Permuted Sequential MNIST | 63.57% | 62.37% | **83.28%** | 67.93% |
| Moving MNIST | 89.31% | 89.45% | **95.07%** | 90.30% |

Dynamic NF 参数量：约 2,799；GRU 参数量：约 20,554。虽然 Dynamic NF 参数更少，但当前实现训练 CPU 时间约为简单递推模型的 3 倍左右，仍低于 GRU 的准确率。

## 解释

目前 Dynamic NF 在 permuted sequence 上略高于简单 ReLU 递推，说明动态节点关系可能保留了一部分顺序组合能力；但它远低于 GRU，说明当前关系场还没有学出有效的长期记忆机制。Moving MNIST 上 Dynamic NF 也只略高于线性/ReLU，不能证明它已经利用了跨帧关系。

注意：第一轮实验曾错误地对 Linear 分支重复执行状态更新，导致输入尺度不一致；该结果已废弃，以上为修正后的正式结果。
