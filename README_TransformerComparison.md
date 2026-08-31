# Transformer 与 Local Electrical NF 比较

这是独立的 Transformer 对照实验，不修改任何 NF 实现。Transformer 使用每个输入特征一个 token：

```text
特征 x_i → shared Linear(1, d_model=8)
加 learnable position embedding
加 CLS token
→ TransformerEncoder（1层、2头、FFN=32）
→ CLS classifier
```

这不是把整个向量作为单个 token；100 维任务的序列长度为 101，因此 Transformer 能够显式建模特征之间的全局关系。

## 公平条件

Transformer、ReLU、GELU 和 Local raw_bounded 使用相同的 seed=0、数据生成和 80/20 划分、batch=128、Adam、lr=0.003、weight decay=1e-4、训练 30 epochs、6000 样本。由于之前困难任务主实验使用了 100 epochs，本表只比较这次统一的 30 epochs，不与 100 epochs 的最高值直接混用。

```powershell
py -3 benchmark_transformer_tasks.py --epochs 30 --n 6000 --batch 128 --seed 0 --d-model 8 --layers 1 --heads 2 --result-tag small30_seed0
```

## 30 轮结果

| 数据集 | ReLU | GELU | Local raw_bounded | Transformer-small |
|---|---:|---:|---:|---:|
| Spiral3 | 100.00% | 99.92% | 100.00% | 100.00% |
| Checkerboard | 56.33% | 53.58% | 52.17% | 53.08% |
| Parity8 | **100.00%** | 52.75% | **100.00%** | 97.17% |
| Noisy Moons100 | 85.83% | 85.92% | 86.42% | **88.67%** |
| Noisy Spiral100 | 90.92% | 90.58% | **95.17%** | **100.00%** |

## 参数量与耗时

| 数据集 | Local NF 参数 | Transformer 参数 | Local NF 秒 | Transformer 秒 |
|---|---:|---:|---:|---:|
| Spiral3 | 582 | 947 | 6.82 | 5.08 |
| Checkerboard | 517 | 938 | 7.22 | 5.51 |
| Parity8 | 901 | 986 | 7.26 | 8.11 |
| Noisy Moons100 | 6789 | 1722 | 7.10 | 70.64 |
| Noisy Spiral100 | 6854 | 1731 | 7.29 | 70.30 |

100 维任务中 Transformer 的序列关系计算使 CPU 训练明显变慢，但参数量反而低于 Local NF，因为 d_model 很小且输入投影共享。

## 结论

1. Transformer 可以解决这些困难任务，而且在 noisy spiral100 上明显超过当前 Local NF。
2. Local NF 在 checkerboard 上反而更好，说明局部传播对某些高频局部结构有帮助；Transformer 的全局关系并不自动适合所有任务。
3. Transformer 在 noisy moons100 和 noisy spiral100 上更强，说明全局 token 交互能更有效地筛选有效维度、建模复杂组合关系。
4. Local NF 在 noisy spiral100 的 30 轮结果为 95.17%，仍然有一定优势于 MLP，但 Transformer 达到 100%，因此当前 Local NF 的优势不是普遍的。
5. 这不是完全参数匹配的最终结论：Transformer-small 参数量约 900～1700，Local NF 约 500～6800。正式结论还应加入参数量匹配 Transformer 和相同推理预算比较。
6. 这组实验也说明“复杂问题”确实能拉开模型差异，但不能只看准确率；Transformer 的高准确率伴随高得多的 100 维任务计算时间。
