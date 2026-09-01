# Local NF 专题分支范围

本分支只把 **Local NF / Local Electrical NF** 作为当前主线。仓库中仍存在从 `main` 继承的 Dynamic NF、BioNeuron、Temporal NF 等历史文件，它们仅为历史兼容，不属于本分支当前研究结论。

## 当前入口

1. `README.md`：分支首页、当前结论、复现命令。
2. `docs/LOCAL_DESIGN_AND_RESULTS.md`：从 `main` 迁移并整合的 Local 核心设计、实验、稳定性和因果结论。
3. `benchmark_minimal_nf_hard_tasks.py`：Minimal Local NF 困难任务。
4. `train_simple_field_ab.py`：Minimal A/B 系列 MNIST 入口，支持 A0/A1/A2/A3、B1/B2/B3 和 circular 变体。

## 当前主线模型

### Minimal Local NF

```text
input -> Linear(256) -> 16×16 field
      -> tanh / optional threshold
      -> shared learnable 3×3 kernel
      -> decay + local incoming
      -> 1-3 steps
      -> Linear readout
```

代表结果：

- `simple_field_a_1step_results_30ep.json`：full MNIST A-1step best 98.01%。
- `hard_task_results/minimal_local_nf_a1_seed0_corrected.json`：A-1step 困难任务。
- `hard_task_results/minimal_local_nf_a23_seed0_corrected.json`：A-2/A-3 困难任务。

### Historical Local Electrical NF

核心历史文件：

- `README_LocalElectricalNF.md`
- `README_LocalElectricalNF_stability.md`
- `local_electrical_nf.py`
- `local_electrical_nf_v2.py`
- `local_electrical_nf_v3.py`
- `train_local_electrical_nf.py`
- `local_electrical_nf_results/`
- `local_electrical_nf_v3_results/`
- `local_electrical_nf_v4_results/`
- `local_electrical_nf_usage_results/`
- `hard_task_results/results_seed0.json`
- `hard_task_results/results_unbounded_seed0.json`

## 当前判断

Local 分支现在不把“MNIST 98.01%”单独当作成功证据。旧 Local Electrical NF 在 Checkerboard / Noisy Spiral100 上出现过更明显优势，而 Minimal A-1step 在 Checkerboard 上退化到接近随机，因此后续要回答的是：

> 优势究竟来自局部拓扑、场状态、多步传播，还是仅来自一次可学习局部非线性混合？

下一阶段优先做多 seed、step 数曲线、fixed/random/learnable kernel、拓扑置换和 parameter-matched baseline。
