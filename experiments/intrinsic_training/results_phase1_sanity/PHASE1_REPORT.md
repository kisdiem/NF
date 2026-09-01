# Phase 1：神经元内在属性训练报告

结果目录：`experiments\intrinsic_training\results_phase1_sanity`

本报告只汇总 E0（联合 BP）、E1（内在属性相对学习率）和 E2（连接/内在属性交替优化）。所有数值均保留完成的 seed，不以单次最高值替代均值。

## bio_neuron

| Strategy | Seeds | Best test accuracy | Final test accuracy | Intrinsic max Δ | Time/run |
|---|---:|---:|---:|---:|---:|
| E0 joint | 1 | 0.1790 ± 0.0000 | 0.1790 ± 0.0000 | 0.012021 | 0.9s |
| E1 lr×0.1 | 1 | 0.1794 ± 0.0000 | 0.1794 ± 0.0000 | 0.0012023 | 1.0s |
| E2 5:1 | 1 | 0.1797 ± 0.0000 | 0.1797 ± 0.0000 | 0 | 0.9s |

当前完成 run 中，按 final accuracy 均值最高的是 **E2 5:1**。

## directional_rect_v4

| Strategy | Seeds | Best test accuracy | Final test accuracy | Intrinsic max Δ | Time/run |
|---|---:|---:|---:|---:|---:|
| E0 joint | 1 | 0.1438 ± 0.0000 | 0.1438 ± 0.0000 | 0.012007 | 1.0s |
| E1 lr×0.1 | 1 | 0.1433 ± 0.0000 | 0.1433 ± 0.0000 | 0.001201 | 1.0s |
| E2 5:1 | 1 | 0.1429 ± 0.0000 | 0.1429 ± 0.0000 | 0 | 1.0s |

当前完成 run 中，按 final accuracy 均值最高的是 **E0 joint**。

## discrete_nf_v3

| Strategy | Seeds | Best test accuracy | Final test accuracy | Intrinsic max Δ | Time/run |
|---|---:|---:|---:|---:|---:|
| E0 joint | 1 | 0.1016 ± 0.0000 | 0.1016 ± 0.0000 | 0.0039982 | 1.4s |
| E1 lr×0.1 | 1 | 0.1014 ± 0.0000 | 0.1014 ± 0.0000 | 0.00039983 | 1.4s |
| E2 5:1 | 1 | 0.1013 ± 0.0000 | 0.1013 ± 0.0000 | 0 | 1.4s |

当前完成 run 中，按 final accuracy 均值最高的是 **E0 joint**。

## dynamic_nf

| Strategy | Seeds | Best test accuracy | Final test accuracy | Intrinsic max Δ | Time/run |
|---|---:|---:|---:|---:|---:|
| E0 joint | 1 | 0.4369 ± 0.0000 | 0.4369 ± 0.0000 | 0.010059 | 0.9s |
| E1 lr×0.1 | 1 | 0.4403 ± 0.0000 | 0.4403 ± 0.0000 | 0.0010061 | 0.9s |
| E2 5:1 | 1 | 0.4403 ± 0.0000 | 0.4403 ± 0.0000 | 0 | 0.9s |

当前完成 run 中，按 final accuracy 均值最高的是 **E2 5:1**。

## hierarchical_nf

| Strategy | Seeds | Best test accuracy | Final test accuracy | Intrinsic max Δ | Time/run |
|---|---:|---:|---:|---:|---:|
| E0 joint | 1 | 0.6930 ± 0.0000 | 0.6930 ± 0.0000 | 0.0098774 | 1.0s |
| E1 lr×0.1 | 1 | 0.6956 ± 0.0000 | 0.6956 ± 0.0000 | 0.0010147 | 1.0s |
| E2 5:1 | 1 | 0.6950 ± 0.0000 | 0.6950 ± 0.0000 | 0 | 1.0s |

当前完成 run 中，按 final accuracy 均值最高的是 **E1 lr×0.1**。

## local_electrical_v1

| Strategy | Seeds | Best test accuracy | Final test accuracy | Intrinsic max Δ | Time/run |
|---|---:|---:|---:|---:|---:|
| E0 joint | 1 | 0.5537 ± 0.0000 | 0.5537 ± 0.0000 | 0.012027 | 1.6s |
| E1 lr×0.1 | 1 | 0.5538 ± 0.0000 | 0.5538 ± 0.0000 | 0.0012027 | 1.6s |
| E2 5:1 | 1 | 0.5536 ± 0.0000 | 0.5536 ± 0.0000 | 0 | 1.6s |

当前完成 run 中，按 final accuracy 均值最高的是 **E1 lr×0.1**。

## local_electrical_v2

| Strategy | Seeds | Best test accuracy | Final test accuracy | Intrinsic max Δ | Time/run |
|---|---:|---:|---:|---:|---:|
| E0 joint | 1 | 0.2360 ± 0.0000 | 0.2360 ± 0.0000 | 0.01183 | 1.6s |
| E1 lr×0.1 | 1 | 0.2374 ± 0.0000 | 0.2374 ± 0.0000 | 0.0011821 | 1.5s |
| E2 5:1 | 1 | 0.2371 ± 0.0000 | 0.2371 ± 0.0000 | 0 | 1.6s |

当前完成 run 中，按 final accuracy 均值最高的是 **E1 lr×0.1**。

## local_electrical_v3

| Strategy | Seeds | Best test accuracy | Final test accuracy | Intrinsic max Δ | Time/run |
|---|---:|---:|---:|---:|---:|
| E0 joint | 1 | 0.5295 ± 0.0000 | 0.5295 ± 0.0000 | 0.012021 | 1.6s |
| E1 lr×0.1 | 1 | 0.5294 ± 0.0000 | 0.5294 ± 0.0000 | 0.0012021 | 1.6s |
| E2 5:1 | 1 | 0.5294 ± 0.0000 | 0.5294 ± 0.0000 | 0 | 1.6s |

当前完成 run 中，按 final accuracy 均值最高的是 **E0 joint**。

## minimal_local_nf

| Strategy | Seeds | Best test accuracy | Final test accuracy | Intrinsic max Δ | Time/run |
|---|---:|---:|---:|---:|---:|
| E0 joint | 1 | 0.3767 ± 0.0000 | 0.3767 ± 0.0000 | 0.001089 | 1.5s |
| E1 lr×0.1 | 1 | 0.3767 ± 0.0000 | 0.3767 ± 0.0000 | 0.00010896 | 1.5s |
| E2 5:1 | 1 | 0.3767 ± 0.0000 | 0.3767 ± 0.0000 | 0 | 1.5s |

当前完成 run 中，按 final accuracy 均值最高的是 **E2 5:1**。

## 完整性与解释限制

- 完成的 model/strategy 组合：27。
- 失败 run：0。
- E1 lr×1 是 E0 的实现一致性对照；两者若不同，优先检查 optimizer/parameter registry，而不是解释成生物效应。
- E2 每个 batch 只更新一个参数组，因此它同时改变了更新时间尺度；结论必须和 E1 一起看。
- L/D 等整数属性未包含在 E0–E2 的 Adam 更新中，后续需要离散局部搜索。
