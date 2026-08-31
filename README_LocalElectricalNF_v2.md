# Local Electrical NF v2: Activity-Dependent Local Inhibition

v2 是独立于 v1 的实验版本，未修改 `local_electrical_nf.py`、`train_local_electrical_nf.py` 或 v1 的 `no_inhibition` 结果。

## 改动

v1 中每个节点有固定且可训练的 excitatory/inhibitory sign，容易让场的平均膜电位产生长期偏移。v2 删除 `sign_raw`：每个节点基础释放均为正向兴奋信号，局部活动过强时才产生负反馈抑制。

```text
release = sigmoid((V - theta_eff) / tau) * strength
E = conv2d(release, K_E)
A = conv2d(release, K_I)
I = beta * sigmoid((A-rho)/tau_I) * A
V_next = decay * V + E - I
```

`K_E` 保持 v1 的正交 1.0、对角 0.7；`K_I` 是归一化的 8 邻居 kernel。`rho` 和 `beta` 是共享可学习标量，不是每节点独立参数。

refractory 作为独立 variant：

```text
R_next = lambda_R * R + release_gate
theta_eff = theta + gamma * R
```

## 运行

```powershell
py -3 train_local_electrical_nf_v2.py --epochs 20 --subset 5000 --batch 128 --steps 4 --device cpu --result-tag seed0
```

结果与热图在 `local_electrical_nf_v2_results/`。本版本仍只使用 `[B,1,8,8]` 的局部 `conv2d`，没有 Q/K、attention、NxN relation 或 GNN。

## 20 轮结果

MNIST subset=5000，full test=10000，batch=128，hidden=64，Adam，lr=0.003，weight decay=1e-4，seed=0，CPU。

| 版本 | best test | final test | 参数量 | CPU 秒 |
|---|---:|---:|---:|---:|
| ReLU | 93.16% | 93.16% | 50,890 | 151.56 |
| GELU | 93.06% | 93.06% | 50,890 | 150.42 |
| no-inhibition v1 | **93.47%** | 93.44% | 51,146 | 181.62 |
| dynamic inhibition | 92.97% | 92.82% | 51,085 | 227.48 |
| dynamic inhibition + refractory | 92.93% | 92.24% | 51,085 | 243.08 |
| dynamic inhibition, 1 step | 92.17% | 92.04% | 51,085 | 194.66 |
| refractory only | 93.32% | 93.19% | 51,085 | 211.39 |

## 动力学诊断

最后一个测试 batch 的诊断如下：

| 版本 | activation rate 各步 | state change 各步 | membrane mean 各步 |
|---|---|---|---|
| no-inhibition v1 | 0.505, 0.679, 0.828, 0.925 | 0.637, 0.747, 1.074, 1.484 | -0.603, 0.142, 1.216, 2.700 |
| dynamic inhibition | 0.127, 0.191, 0.314, 0.484 | 0.712, 0.770, 1.024, 1.388 | -0.936, -0.170, 0.854, 2.242 |
| + refractory | 0.144, 0.200, 0.318, 0.483 | 0.729, 0.773, 1.012, 1.363 | -0.765, 0.002, 1.013, 2.376 |
| refractory only | 0.139, 0.210, 0.360, 0.545 | 0.663, 0.773, 1.133, 1.553 | -0.543, 0.226, 1.358, 2.910 |

动态抑制把激活率显著压低，但 `state_change` 没有随时间下降，膜电位仍向正方向累积。因此它抑制了“节点是否释放”，却没有抵消正向兴奋传播带来的总能量增长。当前 `beta`/`rho` 的学习没有形成足够强的稳态负反馈，不能宣称 v2 已解决正反馈扩散。

## 结论

1. 动态局部抑制比 v1 的固定 inhibitory sign 更符合设计目标：不再固定制造负膜电位漂移，且活动率更低；但本实验没有证明它更稳定，因为 state change 仍增长。
2. refractory 没有进一步改善稳定性，且准确率从 92.97% 降到 92.93%，最终准确率降到 92.24%。
3. 最高准确率仍为 v1 的 no-inhibition：93.47%。
4. 最稳定的准确率表现是 v1 no-inhibition 和 refractory-only，但从动力学数值看都仍有逐步增长趋势。
5. 多步传播优于 1 step：动态抑制版本 92.97% 对 92.17%。
6. v2 完全没有 attention、QK 或 NxN 计算，复杂度仍是 `O(N*k*T)`，其中 `N=64,k=8`。动态抑制每一步额外增加一次局部卷积，局部传播估计为 `8*8*9*2*2=2304` FLOPs/step。
7. 目前不建议扩大网格或增加 steps。首先需要让 excitation 与 inhibition 在同一量纲上形成真正的负反馈，否则增加空间范围或时间步只会放大累计偏移。
