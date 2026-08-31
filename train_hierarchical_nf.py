"""Benchmark the 8+8 hierarchical Dynamic NF beside the flat Dynamic NF."""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamic_nf import DynamicNFMLP
from hierarchical_nf import HierarchicalDynamicNFMLP
from train_mnist import load_data


class MLP(nn.Module):
    def __init__(self, activation=None):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(784, 64), nn.Linear(64, 10)
        self.activation = activation

    def forward(self, x, return_hidden=False):
        h = self.fc1(x.flatten(1))
        if self.activation is not None:
            h = self.activation(h)
        y = self.fc2(h)
        return (y, h) if return_hidden else y


def make_model(kind, args):
    if kind == "linear": return MLP()
    if kind == "relu": return MLP(F.relu)
    if kind == "gelu": return MLP(F.gelu)
    if kind == "flat_dynamic":
        return DynamicNFMLP(hidden=64, n_nodes=16, node_dim=4,
                            branches=4, steps=args.steps,
                            relation_gain_init=0.1, temperature=1.0,
                            norm=True)
    return HierarchicalDynamicNFMLP(
        hidden=64, n_nodes=8, node_dim=4, branches=4, steps=args.steps,
        relation_gain_init=0.1, temperature=1.0,
        layer2_internal=kind != "hier_no_l2", feedback=kind == "hier_feedback",
        feedback_gain_init=0.02, norm=True)


def save_diagnostics(field, name, out_dir):
    with open(os.path.join(out_dir, name + "_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(field.last_diagnostics, f, indent=2, ensure_ascii=False)
    try:
        import matplotlib.pyplot as plt
        for t, pack in enumerate(field.last_relations):
            if pack.ndim == 3:
                plt.figure(figsize=(3, 2.8)); plt.imshow(pack[0].cpu(), cmap="coolwarm")
                plt.colorbar(); plt.title(f"{name} relation t={t}")
                plt.xlabel("target"); plt.ylabel("source"); plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"{name}_relation_t{t}.png"), dpi=130); plt.close()
                continue
            for k, label in enumerate(("r11", "r12", "r22", "r21")):
                if label == "r21" and not field.feedback:
                    continue
                plt.figure(figsize=(3, 2.8)); plt.imshow(pack[0, k].cpu(), cmap="coolwarm")
                plt.colorbar(); plt.title(f"{name} {label} t={t}")
                plt.xlabel("target"); plt.ylabel("source"); plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"{name}_{label}_t{t}.png"), dpi=130); plt.close()
        d = field.last_diagnostics
        plt.figure(figsize=(5, 3)); plt.plot(d["state_change"], marker="o", label="state")
        plt.plot(d["relation_change"], marker="s", label="relation")
        plt.legend(); plt.xlabel("step"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, name + "_state_change.png"), dpi=130); plt.close()
    except Exception as exc:
        print("visualization skipped:", exc)


def run(kind, args, train_loader, test_loader, device, out_dir):
    torch.manual_seed(args.seed)
    model = make_model(kind, args).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history, start_time = [], time.process_time()
    for ep in range(args.epochs):
        model.train(); loss_sum = correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x); loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True); loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            loss_sum += loss.item() * y.numel(); correct += (logits.argmax(1) == y).sum().item(); total += y.numel()
        model.eval(); test_correct = test_total = test_loss = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device); logits = model(x)
                test_loss += F.cross_entropy(logits, y).item() * y.numel()
                test_correct += (logits.argmax(1) == y).sum().item(); test_total += y.numel()
        row = {"epoch": ep + 1, "train_loss": loss_sum / total,
               "train_acc": correct / total, "test_loss": test_loss / test_total,
               "test_acc": test_correct / test_total, "grad_norm": float(grad)}
        history.append(row)
        print(f"{kind:15s} ep{ep+1:02d}/{args.epochs} train={row['train_acc']:.4f} test={row['test_acc']:.4f}")
    result = {"model": kind, "history": history,
              "best_test_acc": max(x["test_acc"] for x in history),
              "final_test_acc": history[-1]["test_acc"],
              "parameters": sum(p.numel() for p in model.parameters()),
              "cpu_seconds": time.process_time() - start_time}
    if hasattr(model, "field"):
        result["diagnostics"] = model.field.last_diagnostics
        result["field_report"] = model.field.parameter_report()
        save_diagnostics(model.field, kind, out_dir)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--subset", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--models", default="linear,relu,gelu,flat_dynamic,hier,hier_no_l2,hier_feedback")
    ap.add_argument("--data-root", default="data/mnist")
    args = ap.parse_args()
    device = torch.device(args.device); out_dir = "hierarchical_nf_results"; os.makedirs(out_dir, exist_ok=True)
    train_loader, test_loader = load_data(args.data_root, args.batch, args.subset)
    results = [run(k, args, train_loader, test_loader, device, out_dir)
               for k in args.models.split(",")]
    with open(os.path.join(out_dir, f"results_seed{args.seed}.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nSUMMARY")
    for r in results:
        print(f"{r['model']:15s} best={r['best_test_acc']:.4f} final={r['final_test_acc']:.4f} params={r['parameters']}")
    print("artifacts:", os.path.abspath(out_dir))


if __name__ == "__main__":
    main()
