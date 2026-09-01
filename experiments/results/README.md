# 实验结果归档规范

本目录用于保存后续所有实验结果。每次实验必须建立一个新的时间目录，禁止覆盖已有 JSON、日志或图表。

## 目录格式

```text
experiments/results/
└── YYYY-MM-DD_HH-mm-ss_<experiment-name>/
    ├── config.json
    ├── summary.json
    ├── history.csv              # 可选
    ├── stdout.log               # 可选
    ├── diagnostics/             # 可选
    └── README.md                # 可选
```

时间使用本地 Asia/Shanghai 时间，名称只使用字母、数字、下划线和连字符。例如：

```text
experiments/results/2026-09-01_21-35-12_bio_mlp_complex_seed0/
```

## 规则

1. 每次启动实验时生成一次时间戳，并在 `config.json` 中记录完整命令、分支、commit、seed、数据集和超参数。
2. 同一次批量实验使用同一个根时间目录，下面按模型或数据集分文件夹。
3. 结果文件中同时记录 `best_test_acc`、`final_test_acc`、耗时、参数量和设备信息。
4. 旧结果只读保留，不重命名、不覆盖。
5. 代码改动和结果归档可以分别提交；结果分支只负责归档可复现实验产物。

推荐批量目录：

```text
experiments/results/YYYY-MM-DD_HH-mm-ss_batch-name/
├── local/
│   └── task_results.json
├── bio/
│   └── task_results.json
└── comparison.csv
```

可以使用仓库根目录的工具创建新目录：

```powershell
python experiments/results/new_run.py --name bio_mlp_complex_seed0
```

