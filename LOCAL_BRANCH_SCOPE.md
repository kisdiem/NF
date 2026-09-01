# Local NF 专题分支

本分支用于集中维护 Local NF / Local Electrical NF 研究，不改变 BioNeuron 实验。

## 当前重点

- `Minimal Local NF A-1step`：256 个场节点、16×16 网格、一次 3×3 局部传播。
- `Local Electrical NF v1/v2/v3/v4`：膜电位、阈值、衰减、强度和动态局部抑制实验。

## 主要历史结果

- `simple_field_a_1step_results_30ep.json`：MNIST 最佳约 98.01%。
- `hard_task_results/`：复杂合成任务及局部场对照。
- `local_electrical_nf_*_results/`：Local Electrical NF 各版本结果。

## 本次整理新增

- `benchmark_minimal_nf_hard_tasks.py`：用 A-1step 测试复杂任务。
- `hard_task_results/minimal_local_nf_a1_seed0_corrected.json`：A-1step 基准。
- `hard_task_results/minimal_local_nf_a23_seed0_corrected.json`：A-2step/3step 对照。

所有结果均保留原始配置和训练曲线，不覆盖旧结果。
