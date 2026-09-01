# BioNeuron：设计与核心证据

## 1. 研究假设

普通点神经元可写成 `y = phi(w^T x)`；BioNeuron 更接近 `y = phi(sum_r g_r(w_r^T x))`。`r` 是树突分支，每个分支先进行独立非线性处理，再由胞体整合。

当前实现还包含兴奋/抑制、短时 trace、膜电位、动态阈值和 adaptation，但这些机制不默认被视为必要贡献。

核心实现：`models/bio_neuron.py`。

## 2. 关键结果

保留原始结果：

- `results/xor_seed0.json`
- `results/circles_seed0.json`
- `results/moons_seed0.json`
- `results/mnist_seed0_20.json`
- `results/mnist_seed1_20.json`
- `results/mnist_seed0_30.json`
- `results/mnist_seed1_30.json`

合成任务：XOR/Circles Bio 100%，Two Moons 100%。这些只证明 Bio 具有非线性表达能力，不证明优于 ReLU/GELU。

MNIST subset=5000、20 epochs：Bio seed 0 best 约 93.3%，seed 1 best 约 93.7%。对应 ReLU/GELU 约 93% 左右，但 Bio 参数量约 1.66×，因此当前没有参数效率优势证据。

## 3. 关键消融

XOR：

| Variant | Accuracy |
|---|---:|
| Full Bio | 100.0% |
| 1 branch | 72.3% |
| no temporal | 100.0% |
| no inhibition | 100.0% |
| fixed threshold | 100.0% |
| one step | 75.0% |

当前最有价值的机制假设是多树突分支；时间、抑制和动态阈值在 XOR 上没有显示独立必要性。由于 4 branches 同时增加参数量，必须通过 parameter-matched 和固定总参数的 branch sweep 才能确认结构贡献。

## 4. 失败诊断

第一版 MNIST 曾出现胞体膜电位约 -5.18、activation 接近 0、dead-neuron ratio=1 的负饱和。修正兴奋/抑制初始化并把树突贡献由求和改为平均后，性能恢复到约 92.6% 以上。这说明内部动力学确实影响训练，但不等于这些动力学本身带来泛化优势。

## 5. 下一阶段判据

继续 Bio 的必要条件：

- parameter-matched 条件下，多 seed 稳定优于普通 MLP；
- branch 数量消融能显示结构性规律，而不是单纯参数增加；
- 至少一个困难组合任务上出现稳定优势；
- 去掉关键树突机制后优势明确消失。

若最终仍只是 93.x% 间的小幅随机波动，则应结束该方向，而不是继续增加生物细节。
