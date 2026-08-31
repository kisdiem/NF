# Transformer 与无限制 NF 对比

本轮在同一数据划分、seed=0、6000 个样本、batch=128、30 epochs、Adam(lr=0.003) 下比较：

- ReLU MLP
- GELU MLP
- Local NF `raw_bounded`
- 无限制 Local NF `raw_unbounded`
- 小型 feature-token Transformer（`d_model=8`、1 层、2 heads）

`raw_unbounded` 是当前 Local Electrical NF 的无限制传播版本：保留 8×8 局部电传播和膜电位状态，但不对局部传播核做有界归一化，并关闭动态抑制。它不使用 Q/K、attention 或 NxN 关系矩阵。

## 结果

| 任务 | ReLU | GELU | Local NF | 无限��� NF | Transformer |
|---|---:|---:|---:|---:|---:|
| spiral3 | 100.00% | 99.92% | 100.00% | 100.00% | 100.00% |
| checkerboard | 56.33% | 53.58% | 52.17% | 54.83% | 53.08% |
| parity8 | 100.00% | 52.75% | 100.00% | 100.00% | 97.17% |
| noisy_moons100 | 85.83% | 85.92% | 86.42% | 85.17% | 88.67% |
| noisy_spiral100 | 90.92% | 90.58% | 95.17% | 91.83% | 100.00% |

表中为每个模型 30 轮内的最高测试准确率；完整逐轮记录见：
`transformer_task_results/results_with_unrestricted30_seed0.json`。

## 训练时间与参数量

| 任务 | 有界 NF 参数/秒 | 无限��� NF 参数/秒 | Transformer 参数/秒 |
|---|---:|---:|---:|
| spiral3 | 582 / 7.28 | 582 / 6.67 | 947 / 5.42 |
| checkerboard | 517 / 7.34 | 517 / 6.34 | 938 / 4.62 |
| parity8 | 901 / 6.73 | 901 / 6.15 | 986 / 7.63 |
| noisy_moons100 | 6789 / 6.82 | 6789 / 5.78 | 1722 / 66.74 |
| noisy_spiral100 | 6854 / 6.32 | 6854 / 5.87 | 1731 / 63.78 |

## 结论

无限制 NF 并没有普遍优于有界 NF：checkerboard 略高，但 noisy_spiral100 明显低于有界 NF，说明取消约束后并未自动带来更强的有效传播，反而可能牺牲稳定的特征形成。Transformer 在两个 100 维噪声任务上最高，但 CPU 训练时间显著更长；在 checkerboard 上不占优势。

运行命令：

```powershell
py -3 benchmark_transformer_tasks.py --epochs 30 --n 6000 --batch 128 --seed 0 --d-model 8 --layers 1 --heads 2 --result-tag with_unrestricted30_seed0
```
