"""Create an objective Markdown report from completed Phase-1 runs."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def label(config):
    if config["strategy"] == "E1": return f"E1 lr×{config['intrinsic_ratio']:g}"
    if config["strategy"] == "E2": return f"E2 {config['alternating']}"
    return "E0 joint"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--output", default=None)
    args = p.parse_args(); root = Path(args.results)
    groups = defaultdict(list); failed = []
    for path in root.glob("*/*/metrics.json"):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
        if metrics["status"] != "complete":
            failed.append((str(path.parent), metrics["error"])); continue
        groups[(config["model"], label(config))].append((config, metrics))

    lines = ["# Phase 1：神经元内在属性训练报告", "",
             f"结果目录：`{root}`", "",
             "本报告只汇总 E0（联合 BP）、E1（内在属性相对学习率）和 E2（连接/内在属性交替优化）。所有数值均保留完成的 seed，不以单次最高值替代均值。", ""]
    for model in sorted({key[0] for key in groups}):
        lines += [f"## {model}", "",
                  "| Strategy | Seeds | Best test accuracy | Final test accuracy | Intrinsic max Δ | Time/run |",
                  "|---|---:|---:|---:|---:|---:|"]
        model_rows = []
        for (key_model, strategy), runs in groups.items():
            if key_model != model: continue
            best = np.array([r[1]["best_test_acc"] for r in runs], dtype=float)
            final = np.array([r[1]["final_test_acc"] for r in runs], dtype=float)
            delta = np.array([r[1]["max_parameter_delta"]["intrinsic"] for r in runs], dtype=float)
            seconds = np.array([r[1]["elapsed_seconds"] for r in runs], dtype=float)
            model_rows.append((float(final.mean()), strategy))
            lines.append(f"| {strategy} | {len(runs)} | {best.mean():.4f} ± {best.std():.4f} | "
                         f"{final.mean():.4f} ± {final.std():.4f} | {delta.mean():.5g} | {seconds.mean():.1f}s |")
        lines += ["", f"当前完成 run 中，按 final accuracy 均值最高的是 **{max(model_rows)[1]}**。"
                  if model_rows else "尚无完成结果。", ""]
    lines += ["## 完整性与解释限制", "",
              f"- 完成的 model/strategy 组合：{len(groups)}。",
              f"- 失败 run：{len(failed)}。",
              "- E1 lr×1 是 E0 的实现一致性对照；两者若不同，优先检查 optimizer/parameter registry，而不是解释成生物效应。",
              "- E2 每个 batch 只更新一个参数组，因此它同时改变了更新时间尺度；结论必须和 E1 一起看。",
              "- L/D 等整数属性未包含在 E0–E2 的 Adam 更新中，后续需要离散局部搜索。", ""]
    if failed:
        lines += ["## 失败记录", ""] + [f"- `{path}`：{error}" for path, error in failed] + [""]
    output = Path(args.output) if args.output else root / "PHASE1_REPORT.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
