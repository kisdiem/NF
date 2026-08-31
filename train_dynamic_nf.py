"""Benchmark DynamicNeuralField and its ablations on the existing MNIST setup."""
import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamic_nf import DynamicNFMLP
from train_mnist import load_data


class MLP(nn.Module):
    def __init__(self, hidden=64, activation=None):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 10)
        self.activation = activation

    def forward(self, x, return_hidden=False):
        h = self.fc1(x.flatten(1))
        if self.activation is not None:
            h = self.activation(h)
        y = self.fc2(h)
        return (y, h) if return_hidden else y


def make_model(kind, args):
    if kind == "linear":
        return MLP(args.hidden)
    if kind == "relu":
        return MLP(args.hidden, F.relu)
    if kind == "gelu":
        return MLP(args.hidden, F.gelu)
    if kind == "matched_mlp":
        return MLP(args.matched_hidden, F.relu)
    common = dict(n_nodes=16, node_dim=4, branches=args.branches,
                  steps=args.steps, relation_gain_init=args.relation_gain,
                  temperature=args.temperature, norm=not args.no_norm,
                  dynamic_relation=True, state_persistence=True,
                  state_gate=True, allow_feedback=True, local_branches=True)
    if kind == "fixed_relation":
        common["dynamic_relation"] = False
    elif kind == "one_step":
        common["steps"] = 1
    elif kind == "no_feedback":
        common["allow_feedback"] = False
    elif kind == "no_state":
        common["state_persistence"] = False
    elif kind == "no_branches":
        common["local_branches"] = False
    return DynamicNFMLP(hidden=64, **common)


def save_diagnostics(field, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, prefix + "_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(field.last_diagnostics, f, indent=2, ensure_ascii=False)
    try:
        import matplotlib.pyplot as plt
        for i, relation in enumerate(field.last_relations):
            plt.figure(figsize=(4, 3.5))
            plt.imshow(relation[0].cpu(), cmap="coolwarm", vmin=-field.relation_gain.item(),
                       vmax=field.relation_gain.item())
            plt.colorbar(); plt.title(f"{prefix} relation t={i}")
            plt.xlabel("target j"); plt.ylabel("source i")
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{prefix}_relation_t{i}.png"), dpi=140); plt.close()
        d = field.last_diagnostics
        plt.figure(figsize=(5, 3.5)); plt.plot(d["state_change"], marker="o", label="state change")
        plt.plot(d["relation_change"], marker="s", label="relation change")
        plt.xlabel("step"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, prefix + "_state_change.png"), dpi=140); plt.close()
    except Exception as exc:
        print(f"visualization skipped: {exc}")


def run(kind, args, train_loader, test_loader, device, out_dir):
    torch.manual_seed(args.seed)
    model = make_model(kind, args).to(device)
    target_params = sum(p.numel() for p in model.parameters())
    if kind == "dynamic":
        args.matched_hidden = max(1, round((target_params - 10) / 795))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    start_time = time.perf_counter()

    def evaluate():
        model.eval(); correct = total = loss_sum = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss_sum += F.cross_entropy(logits, y).item() * y.numel()
                correct += (logits.argmax(1) == y).sum().item(); total += y.numel()
        return loss_sum / total, correct / total

    for ep in range(args.epochs):
        model.train(); train_loss = train_correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x); loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True); loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * y.numel()
            train_correct += (logits.argmax(1) == y).sum().item(); total += y.numel()
        test_loss, test_acc = evaluate()
        row = {"epoch": ep + 1, "train_loss": train_loss / total,
               "test_loss": test_loss, "train_acc": train_correct / total,
               "test_acc": test_acc, "gradient_norm": float(grad_norm)}
        history.append(row)
        if ep == args.epochs - 1:
            if hasattr(model, "field"):
                model.field.last_diagnostics["gradient_norm"] = float(grad_norm)
        print(f"{kind:15s} ep{ep + 1:02d}/{args.epochs} "
              f"train={row['train_acc']:.4f} test={test_acc:.4f} loss={row['test_loss']:.4f}")

    result = {"model": kind, "history": history,
              "best_test_acc": max(x["test_acc"] for x in history),
              "final_test_acc": history[-1]["test_acc"],
              "parameters": sum(p.numel() for p in model.parameters()),
              "seconds": time.perf_counter() - start_time}
    if hasattr(model, "field"):
        result["diagnostics"] = model.field.last_diagnostics
        result["approx_flops_per_sample"] = model.field.parameter_report()["approx_flops_per_step"] * model.field.steps
        save_diagnostics(model.field, out_dir, kind)
    else:
        result["approx_flops_per_sample"] = 2 * sum(p.numel() for p in model.parameters())
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--subset", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--matched-hidden", type=int, default=64)
    ap.add_argument("--branches", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--relation-gain", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--no-norm", action="store_true")
    ap.add_argument("--data-root", default="data/mnist")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    train_loader, test_loader = load_data(args.data_root, args.batch, args.subset)
    out_dir = "dynamic_nf_results"
    os.makedirs(out_dir, exist_ok=True)
    kinds = ["linear", "relu", "gelu", "dynamic", "fixed_relation",
             "one_step", "no_feedback", "no_state", "no_branches"]
    results = [run(k, args, train_loader, test_loader, device, out_dir) for k in kinds]
    # Match an MLP to the dynamic model's parameter count after measuring it.
    dynamic_params = next(r["parameters"] for r in results if r["model"] == "dynamic")
    args.matched_hidden = max(1, round((dynamic_params - 10) / 795))
    matched = run("matched_mlp", args, train_loader, test_loader, device, out_dir)
    results.append(matched)
    with open(os.path.join(out_dir, f"results_seed{args.seed}.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nSUMMARY")
    for r in results:
        print(f"{r['model']:15s} best={r['best_test_acc']:.4f} "
              f"final={r['final_test_acc']:.4f} params={r['parameters']}")
    print(f"artifacts: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
