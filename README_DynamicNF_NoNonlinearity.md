# Dynamic NF：去除非线性函数的对照

本实验区分两种“去掉非线性”：

1. `no_nonlinearity`：去掉 `tanh`、`sigmoid`、`softmax` 和 LayerNorm，但保留状态相关的 `QK^T`。因此关系乘法仍使整体映射成为双线性，不是严格线性。
2. `strict_linear_nf`：使用固定可学习关系矩阵、线性消息和线性状态更新，不依赖输入生成关系，也不使用任何显式非线性。

MNIST 子集 5000、20 轮、batch=128、seed=0：

| 版本 | 最高准确率 | 最终准确率 | 结论 |
|---|---:|---:|---|
| Linear | 90.61% | 89.54% | 线性基线 |
| Dynamic NF | 93.14% | 92.95% | 完整版本 |
| No nonlinearity | 91.73% | 90.79% | QK 双线性交互仍在，但训练不稳定 |
| Strict linear NF | 90.48% | 88.73% | 与 Linear 基本一致 |

`no_nonlinearity` 训练过程中曾出现很大的测试 loss（例如 1259、101、79），原因是去掉有界激活后关系和状态没有足够的数值约束。`strict_linear_nf` 与 Linear 接近，验证了显式非线性是性能提升的重要来源；但当前实验还不能把提升单独归因于动态关系，因为固定关系和 one-step 版本此前更好。
