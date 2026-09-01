# NF / Local Branch

本分支专门维护 **Local NF / Local Electrical NF**。BioNeuron 研究请切换到 `bio` 分支；完整历史与其他探索版本保留在 `main`。

## 当前研究目标

Local 路线研究：在隐藏表示中加入局部邻接、膜电位状态和重复传播，是否能形成普通逐点激活函数没有的能力，并在复杂非线性或高噪声任务中获得稳定收益。

当前分为两代：

```text
Local Electrical NF v1-v4
输入 -> Linear -> 8×8 场 -> 阈值释放/膜电位/局部传播 -> 多步演化 -> 输出

Minimal Local NF
输入 -> Linear(256) -> 16×16 场 -> tanh + 可学习 3×3 kernel -> 1-3 步 -> 输出
```

完整设计与实验结论：[`docs/LOCAL_DESIGN_AND_RESULTS.md`](docs/LOCAL_DESIGN_AND_RESULTS.md)。

## 当前最重要结果

### Minimal Local NF

- Full MNIST，A-1step，30 epochs，seed=0：**98.01% best test**。
- 1-step 在现有 MNIST 比较中优于更多传播步，说明“多步越多越强”并不成立。
- Minimal A-1step 在困难任务上：Spiral3 100.00%，Checkerboard 53.33%，Parity8 78.75%，Noisy Moons100 86.67%，Noisy Spiral100 90.92%。
- 因此极简版本虽然 MNIST 更高，但没有继承旧 Local Electrical NF 在 Checkerboard / Noisy Spiral100 上的优势。

### 历史 Local Electrical NF

MNIST subset=5000、统一 25 epochs、seed=0：

| 模型 | best test |
|---|---:|
| ReLU | 93.27% |
| GELU | 93.19% |
| Local Electrical NF | 92.99% |
| no-inhibition Local NF | **93.47%** |

困难任务的单 seed 历史结果中：

- Checkerboard：ReLU 58.50%，Local raw-bounded 64.75%，raw-unbounded 68.58%。
- Noisy Spiral100：ReLU 91.92%，Local raw-bounded **95.33%**。

这些优势尚需多 seed 和参数匹配验证。

## 关键因果结论

冻结 field 参数时 91.67%，可训练 field 为 91.74%，差异仅 0.07 个百分点；但打乱空间位置会降至 9.59%，置零 field 为 10.32%。

因此目前最准确的结论是：

> 场结构本身确实被模型使用，但“训练出来的场动力学参数”相对固定场变换的额外泛化价值尚未被充分证明。

## 核心文件

```text
LOCAL_BRANCH_SCOPE.md
README_LocalElectricalNF.md
README_LocalElectricalNF_stability.md
README_HardTasks.md
docs/LOCAL_DESIGN_AND_RESULTS.md

train_simple_field_ab.py
benchmark_minimal_nf_hard_tasks.py
local_electrical_nf.py
local_electrical_nf_v2.py
local_electrical_nf_v3.py
train_local_electrical_nf.py

simple_field_a_1step_results_30ep.json
simple_field_a_steps_results_20ep.json
hard_task_results/minimal_local_nf_a1_seed0_corrected.json
hard_task_results/minimal_local_nf_a23_seed0_corrected.json
hard_task_results/results_seed0.json
local_electrical_nf_results/
local_electrical_nf_v3_results/
local_electrical_nf_v4_results/
local_electrical_nf_usage_results/
```

## 复现实验

Minimal MNIST：

```bash
python train_simple_field_ab.py --epochs 30 --batch 128 --variants A_1step --result simple_field_a_1step_repro.json
```

Minimal 困难任务：

```bash
python benchmark_minimal_nf_hard_tasks.py --steps 1 --epochs 100 --seed 0 --lr 0.0003 --result hard_task_results/minimal_local_nf_a1_repro.json
```

A-2/A-3：

```bash
python benchmark_minimal_nf_hard_tasks.py --steps 2,3 --epochs 100 --seed 0 --lr 0.0003 --result hard_task_results/minimal_local_nf_a23_repro.json
```

历史 Local Electrical NF：

```bash
python train_local_electrical_nf.py --epochs 25 --subset 5000 --batch 128 --steps 4
```

## 当前优先级

下一阶段不建议继续堆新机制。优先完成：

1. Checkerboard / Noisy Spiral100 多 seed；
2. `steps=0/1/2/3/4/6/8`；
3. learnable kernel vs fixed/random-fixed kernel；
4. 固定 16×16 邻接 vs 随机邻接 vs 可学习拓扑；
5. parameter-matched MLP、小 CNN、recurrent local conv 对照。

目标是判断真正有价值的是“局部拓扑”“单次局部混合”还是“多步场动力学”，而不是继续默认 Local NF 的所有机制都必要。
