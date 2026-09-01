# Bio-easy：静态图像与无时序逻辑版本

本分支保存当前实验中表现稳定、结构较轻量的 `bio-eazy` 版本。它的定位不是完整的时序神经元模拟，而是一个用于替换 MLP 单一激活函数的、带树突分支局部组合的非线性模块。

## 适用范围

Bio-easy 主要服务于：

- MNIST 等基本静态图像分类；
- XOR、圆形、月牙、棋盘格、Parity 等无时序逻辑任务；
- 输入一次给定、输出一次分类结果的前馈任务。

当前版本不应被解释为已经解决了真正的时序状态记忆或 FSM 执行问题。FSM Sequence 上的多 step 结果只能说明重复的局部非线性演化可能改善表示，不能证明模型已经学会了可泛化的状态转移机制。

## 当前结构

```text
Input
  ↓
Linear(input_dim, 64)
  ↓
BioNeuronLayer(64 neurons, 4 branches)
  ↓
Linear(64, num_classes)
```

每个神经元包含 4 个局部分支。每个分支对输入做独立组合，再由胞体整合。当前默认的 `bio-eazy` 配置为：

```text
branches = 4
temporal = False
membrane_decay = 0
steps = 1
inhibition = True
adaptive_threshold = True
dendrite = soft_threshold
output_mode = mean
```

因此，当前版本的核心有效机制是：

```text
多分支局部组合 + 软阈值 + 兴奋/抑制组合 + 自适应阈值
```

而不是持续膜电位或长时间传播。

## 与 Bio-old 的边界

`bio-old` 仍保留 temporal trace、膜电位衰减和多步状态演化，适合单独研究时序动态，但这些机制在静态任务上可能增加优化难度和计算开销。因此本分支只固定保存轻量的 Bio-easy，不将 Bio-old 的时序结论混入本版本。

## 后续研究分支

后续时序研究应从本分支另开：

```text
bio-temporal
```

该分支专门研究：

- 膜电位跨时间步持续；
- temporal trace 与 refractory state；
- 多步状态演化；
- FSM、Delayed XOR、事件流等时序任务；
- 时序模型与普通 MLP、RNN/GRU 的公平比较。

这样可以把“静态非线性替换实验”和“时序神经元动力学实验”分开，避免用同一个模型同时承担两个不同研究问题。

## 相关代码与结果

- 核心实现：`models/bio_neuron.py`
- Bio 对照脚本：`experiments/results/run_bio_easy_old_logic_suite.py`
- FSM step 对照：`experiments/results/run_bio_fsm_steps.py`
- 当前 FSM 结果：`experiments/results/2026-09-02_00-38-46_bio_fsm_steps_comparison_seeds012/`

本分支的实验结果应优先按静态图像和无时序逻辑任务解读；如果启用 `temporal=True` 或依赖多步膜电位状态，应转移到 `bio-temporal` 分支记录。
