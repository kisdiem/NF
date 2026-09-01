"""Strict conventional MLP baselines on the independent FSM Sequence task."""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from experiments.results.run_bio_easy_old_logic_suite import data


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build(kind, d, hidden, classes):
    if kind == "linear":
        return nn.Sequential(nn.Linear(d, hidden), nn.Linear(hidden, classes))
    act = nn.ReLU() if kind == "relu" else nn.GELU()
    return nn.Sequential(nn.Linear(d, hidden), act, nn.Linear(hidden, classes))


def run(kind, hidden, seed, args):
    seed_all(seed)
    xtr, xte, ytr, yte, classes = data("fsm_sequence", args.n, seed)
    model = build(kind, xtr.shape[1], hidden, classes).to(args.device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_loader = DataLoader(TensorDataset(xtr, ytr), args.batch, shuffle=True)
    test_loader = DataLoader(TensorDataset(xte, yte), 1024, shuffle=False)
    history = []
    best = 0.0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        good = total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(args.device), yb.to(args.device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_sum += loss.item() * yb.numel()
            good += (logits.argmax(1) == yb).sum().item()
            total += yb.numel()
        model.eval()
        test_good = test_total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                logits = model(xb.to(args.device))
                test_good += (logits.argmax(1) == yb.to(args.device)).sum().item()
                test_total += yb.numel()
        row = {"epoch": epoch, "train_loss": loss_sum / total,
               "train_acc": good / total, "test_acc": test_good / test_total}
        history.append(row)
        best = max(best, row["test_acc"])
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"[{kind}-{hidden} seed={seed}] ep={epoch:03d} "
                  f"train={row['train_acc']:.4f} test={row['test_acc']:.4f}", flush=True)
    seconds = time.perf_counter() - start
    return {"task": "fsm_sequence", "variant": f"{kind}-{hidden}", "seed": seed,
            "independent_test_seed": seed + 10000, "parameters": params,
            "seconds": seconds, "seconds_per_epoch": seconds / args.epochs,
            "best_test_acc": best, "final_test_acc": history[-1]["test_acc"],
            "history": history}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--n", type=int, default=6000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--result", required=True)
    args = p.parse_args()
    variants = [("linear", 64), ("relu", 64), ("gelu", 64),
                ("relu", 128), ("gelu", 128), ("relu", 256), ("gelu", 256)]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    output = {"config": vars(args), "results": []}
    os.makedirs(os.path.dirname(os.path.abspath(args.result)), exist_ok=True)
    for kind, hidden in variants:
        for seed in seeds:
            output["results"].append(run(kind, hidden, seed, args))
            with open(args.result, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2)
    print("SUMMARY")
    for kind, hidden in variants:
        rows = [r for r in output["results"] if r["variant"] == f"{kind}-{hidden}"]
        print(f"{kind}-{hidden:3d} mean_best={np.mean([r['best_test_acc'] for r in rows]):.4f} "
              f"std={np.std([r['best_test_acc'] for r in rows], ddof=1):.4f} "
              f"params={rows[0]['parameters']}")


if __name__ == "__main__":
    main()
