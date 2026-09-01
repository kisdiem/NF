"""Prioritized, resumable queue for Phase-1 intrinsic-training experiments."""
import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Run:
    model: str
    strategy: str
    seed: int
    intrinsic_ratio: float = 0.1
    alternating: str = "5:1"

    def command(self, args, output_root):
        command = [
            sys.executable, "-m", "experiments.intrinsic_training.run_phase1",
            "--model", self.model, "--strategy", self.strategy,
            "--seed", str(self.seed), "--epochs", str(args.epochs),
            "--subset", str(args.subset), "--batch", str(args.batch),
            "--output-root", str(output_root), "--data-root", args.data_root,
        ]
        if self.strategy == "E1":
            command += ["--intrinsic-ratio", str(self.intrinsic_ratio)]
        if self.strategy == "E2":
            command += ["--alternating", self.alternating]
        return command

    def run_name(self, args):
        tag = (f"r{self.intrinsic_ratio:g}" if self.strategy == "E1" else
               f"a{self.alternating.replace(':', '_')}" if self.strategy == "E2"
               else "joint")
        return (f"{self.model}__{self.strategy}_{tag}__seed{self.seed}"
                f"__ep{args.epochs}__n{args.subset}")


RATIOS = (1.0, 0.3, 0.1, 0.03, 0.01)
SCHEDULES = ("1:1", "3:1", "5:1", "10:1")
PRIMARY = ("minimal_local_nf", "local_electrical_v1", "local_electrical_v3",
           "directional_rect_v4", "discrete_nf_v3")
SECONDARY = ("bio_neuron", "dynamic_nf")
SANITY_ONLY = ("local_electrical_v2", "hierarchical_nf")


def strategies(model, seed):
    yield Run(model, "E0", seed)
    for ratio in RATIOS:
        yield Run(model, "E1", seed, intrinsic_ratio=ratio)
    for schedule in SCHEDULES:
        yield Run(model, "E2", seed, alternating=schedule)


def build_queue(profile):
    if profile == "sanity":
        return [r for model in PRIMARY + SECONDARY + SANITY_ONLY
                for r in (Run(model, "E0", 0), Run(model, "E1", 0, 0.1),
                          Run(model, "E2", 0, alternating="1:1"))]

    queue = []
    # Highest-value controls first: establish the joint-BP baseline on every
    # important family before spending time sweeping any one strategy.
    for seed in (0, 1):
        queue += [Run(model, "E0", seed) for model in PRIMARY + SECONDARY]
    # Then complete all E1/E2 comparisons for two seeds.
    for seed in (0, 1):
        for model in PRIMARY + SECONDARY:
            queue += list(strategies(model, seed))[1:]
    # A third seed for the five clearest intrinsic-parameter families brings
    # the planned total to 190 runs without diluting the question with wrappers.
    for model in PRIMARY:
        queue += list(strategies(model, 2))
    return queue


def append_jsonl(path, item):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def summarize(output_root):
    rows = []
    for path in output_root.glob("*/*/metrics.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
        rows.append({
            "model": config["model"], "strategy": config["strategy"],
            "intrinsic_ratio": config["intrinsic_ratio"],
            "alternating": config["alternating"], "seed": config["seed"],
            "status": data["status"], "best_test_acc": data["best_test_acc"],
            "final_test_acc": data["final_test_acc"],
            "elapsed_seconds": data["elapsed_seconds"],
            "intrinsic_delta": data["max_parameter_delta"]["intrinsic"],
            "synaptic_delta": data["max_parameter_delta"]["synaptic"],
            "max_inactive_change": data["max_inactive_change"],
            "run_dir": str(path.parent),
        })
    rows.sort(key=lambda x: (x["model"], x["strategy"], x["seed"],
                             x["intrinsic_ratio"], x["alternating"]))
    if rows:
        with (output_root / "aggregate_results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", choices=("sanity", "overnight"), default="sanity")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--max-hours", type=float, default=6.5)
    p.add_argument("--data-root", default="data/mnist")
    p.add_argument("--output-root", default=None)
    args = p.parse_args()
    if args.epochs is None: args.epochs = 1 if args.profile == "sanity" else 10
    if args.subset is None: args.subset = 512 if args.profile == "sanity" else 5000
    if args.output_root is None:
        args.output_root = f"experiments/intrinsic_training/results_phase1_{args.profile}"
    output_root = Path(args.output_root); output_root.mkdir(parents=True, exist_ok=True)
    queue = build_queue(args.profile)
    manifest = {
        "profile": args.profile, "planned_runs": len(queue),
        "epochs": args.epochs, "subset": args.subset, "batch": args.batch,
        "max_hours": args.max_hours,
        "research_order": ["E0 cross-family controls", "E1/E2 two seeds",
                           "third seed on primary intrinsic families"],
    }
    (output_root / "queue_config.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    events = output_root / "queue_events.jsonl"
    start = time.perf_counter(); completed = failed = skipped = 0
    for index, run in enumerate(queue, 1):
        elapsed_hours = (time.perf_counter() - start) / 3600
        if elapsed_hours >= args.max_hours:
            append_jsonl(events, {"event": "time_limit", "next_index": index,
                                  "elapsed_hours": elapsed_hours})
            break
        run_dir = output_root / run.model / run.run_name(args)
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            skipped += 1
            append_jsonl(events, {"event": "skip_complete", "index": index,
                                  "run": run.run_name(args)})
            continue
        command = run.command(args, output_root)
        print(json.dumps({"event": "start", "index": index, "total": len(queue),
                          "run": run.run_name(args)}, ensure_ascii=False), flush=True)
        run_start = time.perf_counter()
        process = subprocess.run(command, text=True, capture_output=True)
        duration = time.perf_counter() - run_start
        log_dir = output_root / "logs"; log_dir.mkdir(exist_ok=True)
        (log_dir / f"{run.run_name(args)}.log").write_text(
            process.stdout + "\nSTDERR\n" + process.stderr, encoding="utf-8")
        event = {"event": "complete" if process.returncode == 0 else "failed",
                 "index": index, "run": run.run_name(args),
                 "returncode": process.returncode, "seconds": duration}
        append_jsonl(events, event); print(json.dumps(event, ensure_ascii=False), flush=True)
        if process.returncode == 0: completed += 1
        else: failed += 1
        summarize(output_root)
    rows = summarize(output_root)
    final = {"event": "queue_finished", "completed_this_invocation": completed,
             "failed_this_invocation": failed, "skipped": skipped,
             "total_result_rows": len(rows),
             "elapsed_hours": (time.perf_counter() - start) / 3600}
    append_jsonl(events, final); print(json.dumps(final, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
