"""Create a timestamped, collision-safe experiment result directory."""
import argparse
import json
import os
import subprocess
from datetime import datetime


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "experiments", "results")


def git_value(args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="short experiment name")
    ap.add_argument("--model", default="", help="optional model name")
    ap.add_argument("--dataset", default="", help="optional dataset name")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--command", default="")
    args = ap.parse_args()
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.name)
    path = os.path.join(RESULTS, f"{stamp}_{safe_name}")
    suffix = 1
    while os.path.exists(path):
        path = os.path.join(RESULTS, f"{stamp}_{safe_name}_{suffix}")
        suffix += 1
    os.makedirs(os.path.join(path, "diagnostics"), exist_ok=False)
    config = {
        "created_at": datetime.now().astimezone().isoformat(),
        "model": args.model,
        "dataset": args.dataset,
        "seed": args.seed,
        "command": args.command,
        "git_branch": git_value(["branch", "--show-current"]),
        "git_commit": git_value(["rev-parse", "HEAD"]),
    }
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    with open(os.path.join(path, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"status": "created", "results": []}, f, indent=2, ensure_ascii=False)
    with open(os.path.join(path, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"# {safe_name}\n\n实验目录创建时间：{config['created_at']}\n")
    print(os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
