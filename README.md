# Local NF

本分支是 Local NF 的干净研究分支，只保留当前两条核心路线、关键对照和能支撑判断的原始结果。其他 NF 历史探索请看 `main`。

## 核心问题

> 局部邻接和状态传播能否形成普通逐点激活没有的能力？如果能，优势来自局部拓扑、一次局部混合，还是多步动力学？

### Minimal Local NF

```text
input -> Linear(256) -> 16×16 field
      -> tanh -> learnable shared 3×3 kernel
      -> decay + local incoming -> 1~3 steps -> readout
```

Full MNIST，A-1step，30 epochs，seed=0：**98.01% best test**。但困难任务上 A-1step 的 Checkerboard 只有 53.33%，Noisy Spiral100 90.92%，说明 MNIST 高分不能单独证明 Local 机制价值。

### Local Electrical NF

```text
input -> Linear(64) -> 8×8 field
      -> threshold release -> local excitation/inhibition
      -> membrane persistence -> repeated local propagation -> readout
```

历史单 seed 结果中，Checkerboard：ReLU 58.50%，raw-bounded 64.75%，raw-unbounded 68.58%；Noisy Spiral100：ReLU 91.92%，raw-bounded **95.33%**。这是目前最值得继续验证的价值线索。

field 因果检查显示：可训练 field 约 91.74%，冻结 field 约 91.67%；但空间打乱约 9.59%，置零约 10.32%。因此目前只能说 **场结构被使用，但场参数学习的额外价值还未证明**。

详细结论见 [`docs/LOCAL_DESIGN_AND_RESULTS.md`](docs/LOCAL_DESIGN_AND_RESULTS.md)。

## 当前目录

```text
train_simple_field_ab.py              # Minimal MNIST / step 对照
benchmark_minimal_nf_hard_tasks.py    # Minimal 困难任务
benchmark_hard_tasks.py               # Electrical vs ReLU/GELU 困难任务
local_electrical_nf.py                # 原始 Electrical 实现
a local_electrical_nf_v3.py           # raw-bounded / raw-unbounded 等稳定版本
diagnose_local_nf_causality_v2.py     # field 冻结与扰动因果检查
results/                               # 仅保留关键 JSON
```

## 复现

```bash
pip install -r requirements.txt
python train_simple_field_ab.py --epochs 30 --variants A_0step_linear_control,A_1step,A_2step,A_3step --result results/new_minimal_mnist.json
python benchmark_minimal_nf_hard_tasks.py --lr 0.0003 --steps 1 --result results/new_minimal_hard.json
python benchmark_hard_tasks.py --epochs 100 --result-tag new
python diagnose_local_nf_causality_v2.py --result results/new_field_causality.json
```

## 下一阶段

只优先做：多 seed；0/1/2/3/4/6/8 step；fixed/random/learnable kernel；拓扑置换；parameter-matched ReLU/GELU/local-mixing 对照。若这些控制实验不能稳定支持 Local，就停止继续堆机制。
