"""Benchmark activity-dependent Local Electrical NF v2 without overwriting v1."""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from local_electrical_nf import LocalElectricalNFMLP
from local_electrical_nf_v2 import LocalElectricalFieldV2, LocalElectricalNFV2MLP
from train_mnist import load_data


class MLP(nn.Module):
    def __init__(self, activation=None):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(784, 64), nn.Linear(64, 10)
        self.activation = activation

    def forward(self, x):
        h = self.fc1(x.flatten(1))
        return self.fc2(self.activation(h) if self.activation else h)


def make_model(kind, args):
    if kind == "relu": return MLP(F.relu)
    if kind == "gelu": return MLP(F.gelu)
    cfg = dict(threshold_init=0.5, strength_init=0.5, decay_init=0.8,
               tau=0.2, dynamic_inhibition=kind != "refractory_only",
               refractory=kind in ("local_dynamic_inhibition_refractory", "refractory_only"),
               rho_init=0.15, beta_init=1.0, tau_inhibition=0.2,
               lambda_r=0.8, gamma_init=0.5)
    steps = 1 if kind == "inhibition_only_one_step" else args.steps
    if kind == "no_inhibition":
        return LocalElectricalNFMLP(8, 8, args.steps, threshold_init=0.5,
                                    strength_init=0.5, decay_init=0.8, tau=0.2,
                                    no_threshold=False, persistence=True, inhibition=False)
    return LocalElectricalNFV2MLP(8, 8, steps, **cfg)


def save_diagnostics(field, out_dir, name):
    with open(os.path.join(out_dir, name + "_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(field.last_diagnostics, f, indent=2, ensure_ascii=False)
    try:
        import matplotlib.pyplot as plt
        inhibitions = getattr(field, "last_inhibitions", [torch.zeros_like(s) for s in field.last_states])
        for t, state in enumerate(field.last_states):
            for data, suffix, title in ((state, "membrane", "membrane"),
                                        (inhibitions[t], "inhibition", "inhibition")):
                plt.figure(figsize=(3, 3)); plt.imshow(data[0, 0].cpu(), cmap="magma")
                plt.colorbar(); plt.title(f"{name} {title} t={t}"); plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"{name}_{suffix}_t{t}.png"), dpi=130); plt.close()
        d = field.last_diagnostics
        plt.figure(figsize=(5, 3))
        plt.plot(d["state_change"], marker="o", label="state change")
        plt.plot(d["activation_rate"], marker="s", label="activation rate")
        plt.plot(d["inhibition_gate_mean"], marker="^", label="inhibition gate")
        plt.legend(); plt.xlabel("step"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, name + "_diagnostics.png"), dpi=130); plt.close()
    except Exception as exc:
        print("visualization skipped:", exc)


def run(kind, args, train_loader, test_loader, device, out_dir):
    torch.manual_seed(args.seed)
    model = make_model(kind, args).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history, start = [], time.process_time()
    for ep in range(args.epochs):
        model.train(); loss_sum = correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device); logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True); loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            loss_sum += loss.item() * y.numel(); correct += (logits.argmax(1) == y).sum().item(); total += y.numel()
        model.eval(); test_loss = test_correct = test_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device); logits = model(x)
                test_loss += F.cross_entropy(logits, y).item() * y.numel()
                test_correct += (logits.argmax(1) == y).sum().item(); test_total += y.numel()
        row = {"epoch": ep + 1, "train_loss": loss_sum / total, "train_acc": correct / total,
               "test_loss": test_loss / test_total, "test_acc": test_correct / test_total,
               "grad_norm": float(grad)}
        history.append(row)
        print(f"{kind:38s} ep{ep+1:02d}/{args.epochs} train={row['train_acc']:.4f} test={row['test_acc']:.4f}")
    result = {"model": kind, "history": history,
              "best_test_acc": max(x["test_acc"] for x in history),
              "final_test_acc": history[-1]["test_acc"],
              "parameters": sum(p.numel() for p in model.parameters()),
              "cpu_seconds": time.process_time() - start}
    if hasattr(model, "field"):
        result["diagnostics"] = model.field.last_diagnostics
        result["field_report"] = model.field.parameter_report()
        save_diagnostics(model.field, out_dir, kind)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--subset", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=128); ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", default="relu,gelu,no_inhibition,local_dynamic_inhibition,local_dynamic_inhibition_refractory,inhibition_only_one_step,refractory_only")
    ap.add_argument("--result-tag", default="seed0"); ap.add_argument("--data-root", default="data/mnist")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); device = torch.device(args.device)
    out_dir = "local_electrical_nf_v2_results"; os.makedirs(out_dir, exist_ok=True)
    train_loader, test_loader = load_data(args.data_root, args.batch, args.subset)
    results = [run(k, args, train_loader, test_loader, device, out_dir) for k in args.models.split(",")]
    path = os.path.join(out_dir, f"results_{args.result_tag}.json")
    with open(path, "w", encoding="utf-8") as f: json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nSUMMARY")
    for r in results:
        print(f"{r['model']:38s} best={r['best_test_acc']:.4f} final={r['final_test_acc']:.4f} params={r['parameters']} time={r['cpu_seconds']:.2f}s")
    print("artifacts:", os.path.abspath(out_dir))


if __name__ == "__main__":
    main()
