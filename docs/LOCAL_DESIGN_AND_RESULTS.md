# Local NF：设计与核心证据

## 1. Minimal Local NF

结构：`Linear -> 16×16 field -> tanh -> shared learnable 3×3 kernel -> decay + incoming -> readout`。

保留结果：

- `results/minimal_mnist_a1_seed0.json`：full MNIST A-1step，best 98.01%。
- `results/minimal_mnist_steps_seed0.json`：step 数早期对照。
- `results/minimal_hard_a1_seed0.json`：A-1step 困难任务。
- `results/minimal_hard_a23_seed0.json`：A-2/A-3 困难任务。

A-1step 困难任务：Spiral3 100.00%，Checkerboard 53.33%，Parity8 78.75%，Noisy Moons100 86.67%，Noisy Spiral100 90.92%。其中 Parity8 只有 256 种可能输入，后续应改成 Parity16/20，当前结果不作为核心成功证据。

## 2. Local Electrical NF

核心更新：

```text
release_i(t) = sigmoid((V_i(t)-T_i)/tau) * S_i
incoming(t)  = local_excitation - local_inhibition
V_i(t+1)     = decay_i * V_i(t) + incoming_i(t)
```

`local_electrical_nf_v3.py` 保留 raw-bounded / raw-unbounded 等关键变体。局部传播使用固定 3×3 邻域，不构造 N×N 全局关系。

关键历史结果：

- `results/electrical_mnist25_seed0.json`：MNIST subset=5000、25 epochs；ReLU 93.27%，完整 Local 92.99%，no-inhibition 93.47%。
- `results/electrical_raw_bounded20_seed0.json`：raw-bounded 稳定性结果。
- `results/electrical_hard_seed0.json`：困难任务 bounded 对照。
- `results/electrical_hard_unbounded_seed0.json`：unbounded 对照。

最值得继续验证的两个结果：

- Checkerboard：ReLU 58.50%，raw-bounded 64.75%，raw-unbounded 68.58%。
- Noisy Spiral100：ReLU 91.92%，raw-bounded 95.33%。

这些目前仍是单 seed，不能作为最终结论。

## 3. 因果证据

- `results/field_freeze_seed0.json`：field 可训练约 91.74%，冻结约 91.67%。
- `results/field_causality_seed0.json`：正常场输出与 shuffle/mean/zero/one-step 等干预。

空间打乱和置零会使性能接近随机，说明场输出确实进入决策；但冻结 field 几乎不掉性能，说明“训练出的场参数”额外贡献很弱。

## 4. 当前结论

Local 研究还没有证明一种普遍优于 ReLU/GELU 的新算子。当前真正需要回答的是：旧 Electrical 在困难边界和噪声任务上的优势，是否能在多 seed、参数匹配和更严格拓扑对照下复现。

下一阶段只测试五个变量：seed、step、kernel 是否学习、拓扑是否固定、参数/FLOPs 是否匹配。