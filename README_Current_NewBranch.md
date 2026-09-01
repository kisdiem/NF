# 当前最新版：Minimal Local Neural Field

本分支 `new` 保存当前最简神经场实验及其结果。

## 当前架构

```text
MNIST 28x28
  -> Linear(784, 256)
  -> reshape [batch, 1, 16, 16]
  -> local field evolution
  -> flatten [batch, 256]
  -> Linear(256, 10)
```

每个场节点不是对应一个原始像素，而是前置线性层对全部 784 个像素的加权组合。

### A：仅膜电位

```text
signal_t = tanh(V_t)
V_(t+1) = decay * V_t + Conv2D(signal_t, K)
```

### B：膜电位 + 阈值

```text
signal_t = sigmoid((V_t - theta) / tau)
V_(t+1) = decay * V_t + Conv2D(signal_t, K)
```

当前默认传播为共享可学习 3x3 局部核、零边界、1--3 个同步时间步；不使用 Q/K、attention、NxN 关系矩阵、inhibition 或 refractory。

## 已完成结果

### 20 轮：A 的时间步比较

见 `simple_field_a_steps_results_20ep.json`。

| 版本 | 最高准确率 |
|---|---:|
| A-1step | 97.80% |
| A-3step | 97.54% |

### 30 轮：A-1step

见 `simple_field_a_1step_results_30ep.json`。

| 版本 | 最高准确率 |
|---|---:|
| A-1step | 98.01% |

纯推理测速脚本为 `benchmark_simple_field_inference.py`。测速时关闭梯度并使用固定 batch，训练时间不计入推理时间。

## 运行

```bash
python train_simple_field_ab.py --epochs 30 --batch 128 --result simple_field_a_1step_results_30ep.json
python benchmark_simple_field_inference.py
```

后续实验默认在 `new` 分支进行，并使用新的结果文件名保存，避免覆盖已有结果。
