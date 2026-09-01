"""Run one reproducible E0/E1/E2 intrinsic-training experiment."""
import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from dynamic_nf import DynamicNFMLP
from hierarchical_nf import HierarchicalDynamicNFMLP
from local_electrical_nf import LocalElectricalNFMLP
from local_electrical_nf_v2 import LocalElectricalNFV2MLP
from local_electrical_nf_v3 import LocalElectricalNFV3MLP
from models.bio_neuron import BioMLP
from models.minimal_local_nf import MinimalLocalNFMLP
from nf_field import NFMLPBlock, RectNFMLPBlock
from training_strategies import (AlternatingBP, JointBP, ParameterGroupBP,
                                 classify_parameters)


MODEL_DEFAULT_LR = {
    "minimal_local_nf": 3e-4,
    "local_electrical_v1": 3e-3,
    "local_electrical_v2": 3e-3,
    "local_electrical_v3": 3e-3,
    "dynamic_nf": 3e-3,
    "hierarchical_nf": 3e-3,
    "bio_neuron": 3e-3,
    "directional_rect_v4": 3e-3,
    "discrete_nf_v3": 1e-3,
}


def optimizer_for(model_key):
    # Preserve the optimizer used by the corresponding original benchmark.
    return torch.optim.AdamW if model_key == "minimal_local_nf" else torch.optim.Adam


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def build_model(key):
    if key == "minimal_local_nf":
        return MinimalLocalNFMLP(hidden=256, steps=1)
    if key == "local_electrical_v1":
        return LocalElectricalNFMLP(8, 8, 4, threshold_init=.5,
                                    strength_init=.5, decay_init=.8,
                                    inhibition=True)
    if key == "local_electrical_v2":
        return LocalElectricalNFV2MLP(8, 8, 4, dynamic_inhibition=True,
                                      refractory=True)
    if key == "local_electrical_v3":
        return LocalElectricalNFV3MLP(8, 8, 4, mode="raw_bounded",
                                      collect_diagnostics=True)
    if key == "dynamic_nf":
        return DynamicNFMLP(steps=4, relation_gain_init=.1)
    if key == "hierarchical_nf":
        return HierarchicalDynamicNFMLP(steps=4)
    if key == "bio_neuron":
        return BioMLP(784, 64, 10, branches=4, steps=3,
                      temporal=True, inhibition=True, adaptive_threshold=True,
                      weight_rank=8, output_mode="mean")
    if key == "directional_rect_v4":
        return nn.Sequential(nn.Flatten(), RectNFMLPBlock(784, 64, 10, {
            "W": 8, "tau_a": .2, "tau_p": 1.0, "residual_alpha": 0.0,
            "gain_init": 1.0, "train_gain": True, "energy_mode": "linear",
            "route_mode": "all", "threshold_init": .5,
        }))
    if key == "discrete_nf_v3":
        return nn.Sequential(nn.Flatten(), NFMLPBlock(784, 64, 10, {
            "H": 8, "W": 8, "K_s": 8, "K_t": 12, "L_max": 4,
            "D_max": 2, "R": 4, "tau": 1.0, "inject_scale": 4.0,
            "surr_scale": 1.0, "read_mode": "potential", "eps_std": 0.0,
        }))
    raise ValueError(key)


def registry_key(key):
    return key


def load_data(root, subset, batch, seed):
    tf = transforms.ToTensor()
    train = datasets.MNIST(root, train=True, download=True, transform=tf)
    test = datasets.MNIST(root, train=False, download=True, transform=tf)
    if subset and subset < len(train):
        # Fixed seed-specific subset, independent of DataLoader shuffle order.
        ids = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed))[:subset]
        train = Subset(train, ids)
    generator = torch.Generator().manual_seed(seed)
    return (DataLoader(train, batch_size=batch, shuffle=True, generator=generator,
                       num_workers=0, pin_memory=torch.cuda.is_available()),
            DataLoader(test, batch_size=1024, shuffle=False, num_workers=0,
                       pin_memory=torch.cuda.is_available()))


def evaluate(model, loader, device):
    model.eval(); loss_sum = correct = total = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x); loss = F.cross_entropy(logits, y)
            loss_sum += float(loss) * y.numel()
            correct += int((logits.argmax(1) == y).sum()); total += y.numel()
    return loss_sum / total, correct / total


def tensor_stats(x):
    x = x.detach().float()
    return {"mean": float(x.mean()), "std": float(x.std(unbiased=False)),
            "min": float(x.min()), "max": float(x.max())}


def effective_tensor(name, p):
    if any(x in name for x in ("decay_raw", "relation_gain_raw")) or (
            "gain" in name and name.endswith("_raw") and "bio." not in name):
        return torch.sigmoid(p)
    if any(x in name for x in ("theta_raw", "strength_raw", "rho_raw",
                               "beta_raw", "gamma_raw", "adaptation_raw",
                               "branch_gain_raw", "soma_gain_raw")):
        return F.softplus(p)
    if "sign_raw" in name:
        return torch.tanh(p)
    if name.endswith("field.theta"):
        return F.softplus(p)
    if name.endswith("field.g_raw"):
        return 2.0 * torch.sigmoid(p)
    if name.endswith("field.T") or name.endswith("field.s_param"):
        return F.softplus(p)
    return p


def collect_parameter_rows(model, inventory, epoch):
    intrinsic = set(inventory.intrinsic_names)
    rows = []
    for name, p in model.named_parameters():
        if name not in intrinsic: continue
        for representation, value in (("raw", p), ("effective", effective_tensor(name, p))):
            rows.append({"epoch": epoch, "parameter": name,
                         "representation": representation, **tensor_stats(value)})
    return rows


def field_of(model):
    if hasattr(model, "field"): return model.field
    for module in model.modules():
        if hasattr(module, "last_diagnostics") and module is not model:
            return module
    return None


def collect_dynamics(model):
    field = field_of(model)
    diag = getattr(field, "last_diagnostics", {}) if field is not None else {}
    clean = {}
    for key, value in diag.items():
        if isinstance(value, (int, float, bool)):
            clean[key] = value
        elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
            clean[key] = value
    return clean


def write_csv(path, rows):
    if not rows: return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys: keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def save_plots(out, history, parameter_rows, dynamics):
    """Small standard plots; failures here must not invalidate a trained run."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [r["epoch"] for r in history]
        for filename, train_key, test_key, ylabel in (
                ("accuracy_curve.png", "train_acc", "test_acc", "accuracy"),
                ("loss_curve.png", "train_loss", "test_loss", "cross entropy")):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(epochs, [r[train_key] for r in history], label="train")
            ax.plot(epochs, [r[test_key] for r in history], label="test")
            ax.set(xlabel="epoch", ylabel=ylabel); ax.grid(alpha=.25); ax.legend()
            fig.tight_layout(); fig.savefig(out / filename, dpi=140); plt.close(fig)

        final_epoch = max((r["epoch"] for r in parameter_rows), default=0)
        final = [r for r in parameter_rows
                 if r["epoch"] == final_epoch and r["representation"] == "effective"]
        if final:
            labels = [r["parameter"].split(".")[-1] for r in final]
            means, stds = [r["mean"] for r in final], [r["std"] for r in final]
            fig, ax = plt.subplots(figsize=(max(6, len(final) * .8), 4))
            ax.bar(range(len(final)), means, yerr=stds, capsize=3)
            ax.set_xticks(range(len(final)), labels, rotation=35, ha="right")
            ax.set_ylabel("effective value (mean ± std)"); ax.grid(axis="y", alpha=.25)
            fig.tight_layout(); fig.savefig(out / "intrinsic_parameter_hist.png", dpi=140)
            plt.close(fig)

        series = {k: v for k, v in dynamics.items()
                  if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v)}
        if series:
            fig, ax = plt.subplots(figsize=(7, 4))
            for key, values in series.items():
                ax.plot(range(1, len(values) + 1), values, marker="o", label=key)
            ax.set(xlabel="internal timestep", ylabel="diagnostic value")
            ax.grid(alpha=.25); ax.legend(fontsize=7, ncol=2)
            fig.tight_layout(); fig.savefig(out / "state_by_timestep.png", dpi=140)
            plt.close(fig)
    except Exception as exc:
        (out / "plot_warning.txt").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(MODEL_DEFAULT_LR), required=True)
    p.add_argument("--strategy", choices=["E0", "E1", "E2"], required=True)
    p.add_argument("--intrinsic-ratio", type=float, default=.1)
    p.add_argument("--alternating", default="5:1")
    p.add_argument("--epochs", type=int, default=2); p.add_argument("--subset", type=int, default=1024)
    p.add_argument("--batch", type=int, default=128); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=None); p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data-root", default="data/mnist")
    p.add_argument("--output-root", default="experiments/intrinsic_training/results")
    args = p.parse_args(); set_seed(args.seed); device = torch.device(args.device)
    lr = args.lr if args.lr is not None else MODEL_DEFAULT_LR[args.model]
    ratio_tag = (f"r{args.intrinsic_ratio:g}" if args.strategy == "E1" else
                 f"a{args.alternating.replace(':', '_')}" if args.strategy == "E2" else "joint")
    run_name = f"{args.model}__{args.strategy}_{ratio_tag}__seed{args.seed}__ep{args.epochs}__n{args.subset}"
    out = Path(args.output_root) / args.model / run_name; out.mkdir(parents=True, exist_ok=False)
    config = vars(args) | {"effective_lr": lr, "run_name": run_name}
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    model = build_model(args.model).to(device)
    inventory = classify_parameters(model, registry_key(args.model))
    syn_steps, int_steps = (int(x) for x in args.alternating.split(":"))
    optimizer_cls = optimizer_for(args.model)
    if args.strategy == "E0":
        strategy = JointBP(model, inventory, lr, args.weight_decay,
                           optimizer_cls=optimizer_cls)
    elif args.strategy == "E1":
        strategy = ParameterGroupBP(model, inventory, lr, args.intrinsic_ratio,
                                    args.weight_decay, optimizer_cls=optimizer_cls)
    else:
        strategy = AlternatingBP(model, inventory, lr, syn_steps, int_steps,
                                 args.weight_decay, optimizer_cls=optimizer_cls)
    train, test = load_data(args.data_root, args.subset, args.batch, args.seed)
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    history, parameter_rows = [], collect_parameter_rows(model, inventory, 0)
    start = time.perf_counter(); failed = None
    try:
        for epoch in range(1, args.epochs + 1):
            model.train(); loss_sum = correct = total = 0; grad_norms = []
            for batch_index, (x, y) in enumerate(train):
                strategy.prepare_batch(batch_index)
                x, y = x.to(device), y.to(device); logits = model(x)
                loss = F.cross_entropy(logits, y)
                if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at epoch {epoch}")
                grad_norms.append(strategy.update(loss, batch_index))
                loss_sum += float(loss.detach()) * y.numel()
                correct += int((logits.argmax(1) == y).sum()); total += y.numel()
            test_loss, test_acc = evaluate(model, test, device)
            dynamics = collect_dynamics(model)
            row = {"epoch": epoch, "train_loss": loss_sum / total,
                   "train_acc": correct / total, "test_loss": test_loss,
                   "test_acc": test_acc, "grad_norm_mean": float(np.mean(grad_norms)),
                   "dynamics_json": json.dumps(dynamics, ensure_ascii=False)}
            history.append(row); parameter_rows.extend(collect_parameter_rows(model, inventory, epoch))
            print(json.dumps({k: v for k, v in row.items() if k != "dynamics_json"}), flush=True)
    except Exception as exc:
        failed = f"{type(exc).__name__}: {exc}"
    finally:
        strategy.finish()
    elapsed = time.perf_counter() - start
    deltas = {}
    for group in ("synaptic", "intrinsic", "other"):
        names = getattr(inventory, f"{group}_names")
        values = [float((dict(model.named_parameters())[n].detach() - initial[n]).abs().max()) for n in names]
        deltas[group] = max(values) if values else 0.0
    metrics = {
        "status": "failed" if failed else "complete", "error": failed,
        "best_test_acc": max((r["test_acc"] for r in history), default=None),
        "final_test_acc": history[-1]["test_acc"] if history else None,
        "elapsed_seconds": elapsed, "parameter_count": sum(p.numel() for p in model.parameters()),
        "parameter_groups": {"synaptic": inventory.synaptic_names,
                             "intrinsic": inventory.intrinsic_names,
                             "other": inventory.other_names,
                             "discrete_intrinsic": inventory.discrete_intrinsic_names},
        "max_parameter_delta": deltas,
        "max_inactive_change": getattr(strategy, "max_inactive_change", None),
        "final_dynamics": collect_dynamics(model),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(out / "history.csv", history); write_csv(out / "parameter_stats.csv", parameter_rows)
    save_plots(out, history, parameter_rows, metrics["final_dynamics"])
    summary = [run_name, f"status={metrics['status']}", f"best_test_acc={metrics['best_test_acc']}",
               f"final_test_acc={metrics['final_test_acc']}", f"elapsed_seconds={elapsed:.3f}",
               f"max_inactive_change={metrics['max_inactive_change']}"]
    (out / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    if failed: raise SystemExit(2)


if __name__ == "__main__": main()
