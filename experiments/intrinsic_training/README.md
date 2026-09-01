# Intrinsic neuronal training experiments

本目录研究连接参数（synaptic weights）与神经元内在属性（threshold、decay、gain、strength 等）是否应使用不同的学习时间尺度。模型定义不被统一改写；分类由外部 registry 提供。

## Phase 1

- `E0`：全部可训练参数使用原 task loss 联合反向传播。
- `E1`：连接参数保持 baseline LR，内在属性使用相对 LR。
- `E2`：连接与内在属性使用独立 optimizer，按 batch 严格交替；非活动组不进入计算图、不 step，因此 Adam state 和 weight decay 都不会偷偷修改它。

运行单项：

```powershell
python -m experiments.intrinsic_training.run_phase1 --model minimal_local_nf --strategy E1 --intrinsic-ratio 0.1 --epochs 20 --subset 20000 --seed 0
```

快速完整性检查：

```powershell
python -m pytest tests/test_intrinsic_training.py -q
python -m experiments.intrinsic_training.run_phase1_queue --profile sanity
```

可续跑的夜间队列（最多 6.5 小时）：

```powershell
python -m experiments.intrinsic_training.run_phase1_queue --profile overnight --epochs 20 --subset 20000 --max-hours 6.5
```

相同输出目录和配置不会被覆盖。队列发现已有 `metrics.json` 会跳过；单个 run 出错会记录日志并继续。每个 run 保存：

```text
config.json
metrics.json
history.csv
parameter_stats.csv
summary.txt
accuracy_curve.png
loss_curve.png
intrinsic_parameter_hist.png
state_by_timestep.png  # 模型提供逐步诊断时
```

生成客观汇总：

```powershell
python -m experiments.intrinsic_training.summarize_phase1 --results experiments/intrinsic_training/results_phase1_overnight
```

参数分类详见仓库根目录的 `neuron_parameter_inventory.md`。
