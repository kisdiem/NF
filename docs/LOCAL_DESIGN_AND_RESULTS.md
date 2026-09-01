# Local NF：设计与核心实验结果

本文件把 `main` 中与 Local / Local Electrical NF 直接相关的核心设计、消融和结果集中到 `local` 分支。原始 JSON、历史 README 和旧版本代码仍保留，便于追溯；这里作为当前研究入口。

## 1. 研究问题

Local NF 研究的核心不是“把 MLP 做得更大”，而是验证：

> 在隐藏表示中引入局部、可重复的状态传播，是否能形成传统逐点激活函数没有的计算能力，同时保持可训练性和较低的连接复杂度？

目前分为两条连续路线：

1. **Local Electrical NF v1-v4**：64 个节点、8×8 场、阈值释放、膜电位持续、多步局部传播，并逐步加入抑制与稳定化。
2. **Minimal Local NF**：把机制压缩到 256 个节点、16×16 场、共享可学习 3×3 kernel、膜电位衰减和 1-3 步传播，用于判断“局部场本身”是否足够。

---

## 2. Local Electrical NF：原始设计

主数据流：

```text
输入
  -> Linear
  -> 8×8 hidden field
  -> membrane potential V
  -> differentiable threshold release
  -> fixed 3×3 local electrical propagation
  -> decay + incoming accumulation
  -> repeat for several steps
  -> Linear readout
```

核心更新近似为：

```text
release_i(t) = sigmoid((V_i(t) - T_i) / tau) * S_i
incoming(t)  = K_local * release(t)
V_i(t+1)     = decay_i * V_i(t) + incoming_i(t)
```

节点可具有阈值、传播强度、衰减和兴奋/抑制属性；局部传播通过 `conv2d` 完成，不构造 N×N 全局关系矩阵。

### 2.1 早期 25 轮公平对照

MNIST subset=5000，seed=0，统一 25 epochs：

| 模型 | best test | final test |
|---|---:|---:|
| Linear | 90.61% | 89.24% |
| ReLU | 93.27% | 93.17% |
| GELU | 93.19% | 93.06% |
| Dynamic NF | 93.14% | 92.97% |
| Local Electrical NF | 92.99% | 92.99% |
| no-inhibition Local NF | **93.47%** | 93.32% |

阶段结论：Local 结构能够训练，但完整版本并未稳定超过 ReLU；no-inhibition 最高也只领先 ReLU 0.20 个百分点，不能据此宣称更优。

原始说明：`README_LocalElectricalNF.md`。

---

## 3. 稳定性实验

v1 no-inhibition 的高准确率伴随明显的正反馈：activation rate 从约 0.505 增至 0.925，state change 从约 0.637 增至 1.484。

主要稳定化结果：

| 版本 | best test | final test | 主要现象 |
|---|---:|---:|---|
| v1 no-inhibition | **93.47%** | 93.44% | 准确率高，但状态持续放大 |
| v2 dynamic inhibition | 92.97% | 92.82% | 降低活动，但仍未阻止累积 |
| v3 normalized | 90.91% | 90.80% | 稳定，但有效信号被明显压缩 |
| v4 full-scale balanced | 93.01% | 91.77% | 稳定性改善，泛化回落 |
| v4 raw-bounded | 93.38% | 93.19% | 当前较平衡方案 |
| v4 raw-strong-bounded | 93.32% | 93.24% | 类似 raw-bounded |

`raw-bounded` 不直接裁剪膜电位，而是在兴奋减抑制后的单步净输入上使用 `tanh`，限制每一步最大更新，同时保留多步传播。

原始说明：`README_LocalElectricalNF_stability.md`。

---

## 4. Field 是否真的被使用

这是 Local 路线最重要的因果性检查之一。

短期冻结实验中：

- 可训练 field：91.74%
- 冻结 field：91.67%

两者只差 0.07 个百分点，说明“field 参数被优化”本身尚未证明带来明显泛化收益。

但前向扰动显示 field 输出又确实不可随意移除：

- 正常 field：91.74%
- 打乱 8×8 空间位置：9.59%
- 只保留空间均值：19.42%
- field 置零：10.32%
- 改成 one-step：59.94%

因此更准确的判断是：

> **空间场变换本身是模型的重要组成部分，但训练后的动态参数相对于“近似固定场变换”的额外价值尚未充分证明。**

这也是后续必须区分“结构归纳偏置”和“可学习动力学贡献”的原因。

---

## 5. 历史困难任务：Local Electrical NF

旧版 Local Electrical NF 在困难合成任务上出现了比 MNIST 更明显的差异：

| 数据集 | ReLU best | Local raw-bounded best | 备注 |
|---|---:|---:|---|
| Spiral3 | 100.00% | 100.00% | 任务过易 |
| Checkerboard | 58.50% | 64.75% | raw-unbounded 达 68.58% |
| Parity8 | 100.00% | 100.00% | 当前生成方式存在大量重复组合，不宜作为强证据 |
| Noisy Moons100 | 85.83% | 86.42% | 优势很小 |
| Noisy Spiral100 | 91.92% | **95.33%** | 单 seed 高约 3.41 个百分点 |

原始结果：`hard_task_results/results_seed0.json`、`hard_task_results/results_unbounded_seed0.json`。

这些结果属于旧版 Local Electrical NF，不能直接当作 Minimal Local NF 的结果。

---

## 6. Minimal Local NF：当前简化路线

当前结构：

```text
input
  -> Linear(d_in, 256)
  -> reshape 16×16
  -> tanh(V)
  -> shared learnable 3×3 local kernel
  -> decay * V + local incoming
  -> 1-3 steps
  -> flatten
  -> Linear(256, classes)
```

MNIST 当前代表结果：

- A-1step，full MNIST，30 epochs，seed=0：**98.01%** best test。
- 20 轮比较中，1-step 高于 3-step，提示“更多传播步数”并不天然更好。

原始结果：

- `simple_field_a_1step_results_30ep.json`
- `simple_field_a_steps_results_20ep.json`
- `simple_field_a_0step_results_30ep.json`

### 6.1 Minimal A-1step 困难任务

在统一 6000 样本、80/20 split、100 epochs、seed=0、lr=3e-4 下：

| 数据集 | A-1step best | A-1step final |
|---|---:|---:|
| Spiral3 | 100.00% | 100.00% |
| Checkerboard | 53.33% | 50.17% |
| Parity8 | 78.75% | 77.25% |
| Noisy Moons100 | 86.67% | 85.92% |
| Noisy Spiral100 | 90.92% | 90.58% |

原始结果：`hard_task_results/minimal_local_nf_a1_seed0_corrected.json`。

这个结果非常重要：**极简 A-1step 并没有继承旧 Local Electrical NF 在 Checkerboard / Noisy Spiral100 上的优势。** 因此“简化后 MNIST 更高”不能解释为 Local 动力学已经被证明；它更可能说明当前 Minimal 版本在 MNIST 上是一个有效局部非线性混合器，但复杂场动力学的独立价值仍待验证。

A-2step / A-3step 结果保存在：`hard_task_results/minimal_local_nf_a23_seed0_corrected.json`。

---

## 7. 当前最可靠的研究结论

1. 局部场结构不是无效旁路；打乱或删除场输出会导致性能崩溃。
2. 旧 Local Electrical NF 在 Checkerboard 和 Noisy Spiral100 上出现过值得继续验证的单 seed 优势。
3. 但 field 参数是否需要学习，目前证据不足；冻结与训练差异极小。
4. 多步传播并非总是有利；在 Minimal 版本中，1-step 在 MNIST 上反而优于更多步。
5. Minimal Local NF 的高 MNIST 准确率不能直接替代旧 Local 的困难任务证据，因为它在 Checkerboard 上几乎退化到随机水平。
6. 因而当前 Local 主问题应从“多步越复杂越强”改为：**什么样的局部拓扑、状态更新和传播深度，能在困难非线性或高噪声任务中产生稳定、不可由普通 MLP 替代的增益？**

---

## 8. 建议的下一组决定性实验

优先级从高到低：

1. Checkerboard、Noisy Spiral100 对 Minimal Local 做 5 seeds，并与同参数量 ReLU/GELU 对照。
2. `steps = 0/1/2/3/4/6/8` 系统曲线，而不是只比较 1 和 3。
3. learnable kernel vs fixed kernel vs random-fixed kernel。
4. 16×16 固定邻接 vs 随机置换邻接 vs 可学习拓扑。
5. parameter-matched MLP / small CNN / recurrent local conv，对抗“这只是一个小卷积”的质疑。
6. 记录准确率之外的状态稳定性、kernel 演化和推理成本。

只有当 Local 在这些对照下仍能稳定保留优势，才能进一步讨论其作为新型计算单元的价值。
