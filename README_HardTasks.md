# Local Electrical NF 困难任务比较

这些任务专门用于避免 XOR/circles/moons 过于简单的问题。统一使用 hidden=64、Adam、lr=0.003、weight decay=1e-4、batch=128、seed=0；每个任务 6000 样本、80/20 train/test，训练 100 epochs。输入标准化只使用训练划分统计量。

运行：

```powershell
py -3 benchmark_hard_tasks.py --epochs 100 --n 6000 --batch 128 --seed 0
```

## 任务

* `spiral3`：三类高曲率螺旋边界。
* `checkerboard`：8×8 高频棋盘格边界。
* `parity8`：8 个二值输入的奇偶组合。
* `noisy_moons100`：2 个有效维度加 98 个高斯干扰维度。
* `noisy_spiral100`：二维三类螺旋加 98 个干扰维度。

## 结果

| 数据集 | ReLU best/final | GELU best/final | Local raw_bounded best/final | Local raw_unbounded best/final | no-inhibition best/final |
|---|---:|---:|---:|---:|---:|
| Spiral3 | 100.00% / 100.00% | 100.00% / 100.00% | 100.00% / 100.00% | 未重跑 | 100.00% / 100.00% |
| Checkerboard | 58.50% / 55.58% | 53.75% / 51.58% | 64.75% / 63.00% | **68.58% / 68.58%** | 54.58% / 51.92% |
| Parity8 | 100.00% / 100.00% | 91.17% / 90.75% | 100.00% / 100.00% | 未重跑 | 100.00% / 100.00% |
| Noisy Moons100 | 85.83% / 82.42% | 85.92% / 81.67% | **86.42% / 82.42%** | 未重跑 | 85.75% / 83.67% |
| Noisy Spiral100 | 91.92% / 91.75% | 91.25% / 91.00% | **95.33% / 95.17%** | 91.83% / 90.25% | 92.25% / 91.00% |

## 解释

Local NF 在 checkerboard 和 noisy spiral100 上取得明显较高的单次最高准确率。后者尤其重要：在 98 个干扰维度存在时，局部动态场比 ReLU 高 3.41 个百分点。不过这是 seed=0 的结果，不能代替多 seed 平均；noisy moons100 的优势很小且最终轮回落，说明泛化稳定性仍不足。

新增的 `raw_unbounded` 不做净输入 `tanh` 限制，也不加入动态 inhibition，用来回答“限制是否本身压低了任务能力”。它在 checkerboard 上达到 68.58%，高于 raw_bounded 的 64.75%；但在 noisy spiral100 上降到 91.83%，远低于 raw_bounded 的 95.33%，说明无界传播可能在高维干扰下积累不稳定。这里的“无界”指传播净输入不做有界化，并不取消 threshold/decay/strength 的基本数值约束。

spiral3 和 parity8 中普通 ReLU 已经达到 100%，不能用来证明 Local NF 优势。checkerboard 的所有模型都不高，说明它确实更难，但也可能存在优化困难。下一步应对 checkerboard/noisy spiral100 跑 seed=1、2、3，并记录均值、标准差和训练曲线，而不是继续增加容易任务。

参数量：二维输入时普通 MLP 322、Local raw_bounded 517；8 维 parity 时分别为 706 和 901；100 维噪声任务分别为 6594 和 6789。Local NF 的参数增量较小，优势不太可能单纯来自参数量暴增，但仍需 parameter-matched MLP 作为正式对照。
