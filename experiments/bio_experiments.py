"""Reproducible BioNeuron experiments.

Examples:
  py -3 experiments/bio_experiments.py --task xor --epochs 300
  py -3 experiments/bio_experiments.py --task all --epochs 200
  py -3 experiments/bio_experiments.py --task mnist --epochs 10 --subset 5000
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.bio_neuron import BioMLP


def synthetic(name, n=1024, seed=0):
    g = torch.Generator().manual_seed(seed)
    if name == "xor":
        base = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        x = base.repeat(n // 4, 1) + 0.08 * torch.randn(n, 2, generator=g)
        y = ((x[:, 0] * x[:, 1]) < 0).long()
    elif name == "circles":
        angles = 2 * torch.pi * torch.rand(n, generator=g)
        y = torch.arange(n) % 2
        radius = torch.where(y == 0, torch.tensor(0.65), torch.tensor(1.55))
        radius = radius + 0.08 * torch.randn(n, generator=g)
        x = torch.stack((radius * angles.cos(), radius * angles.sin()), dim=1)
    elif name == "moons":
        half = n // 2
        t = torch.pi * torch.rand(half, generator=g)
        x0 = torch.stack((t.cos(), t.sin()), dim=1)
        x1 = torch.stack((1 - t.cos(), 0.35 - t.sin()), dim=1)
        x = torch.cat((x0, x1), 0) + 0.10 * torch.randn(n, 2, generator=g)
        y = torch.cat((torch.zeros(half), torch.ones(n - half))).long()
    else:
        raise ValueError(name)
    perm = torch.randperm(n, generator=g)
    return x[perm], y[perm]


class LinearBaseline(nn.Module):
    def __init__(self, d_in, hidden, d_out):
        super().__init__()
        self.hidden = nn.Linear(d_in, hidden)
        self.out = nn.Linear(hidden, d_out)

    def forward(self, x, return_hidden=False):
        h = self.hidden(x.flatten(1))
        y = self.out(h)
        return (y, h) if return_hidden else y


class ActivationBaseline(nn.Module):
    def __init__(self, d_in, hidden, d_out, activation):
        super().__init__()
        self.hidden = nn.Linear(d_in, hidden)
        self.activation = activation
        self.out = nn.Linear(hidden, d_out)

    def forward(self, x, return_hidden=False):
        h = self.activation(self.hidden(x.flatten(1)))
        y = self.out(h)
        return (y, h) if return_hidden else y


def make_model(kind, d_in, hidden, d_out, ablation=None, steps=3,
               hard_spike=False, dendrite="soft_threshold", bio_rank=0,
               bio_output="mean"):
    if kind == "linear":
        return LinearBaseline(d_in, hidden, d_out)
    if kind == "relu":
        return ActivationBaseline(d_in, hidden, d_out, F.relu)
    if kind == "gelu":
        return ActivationBaseline(d_in, hidden, d_out, F.gelu)
    cfg = dict(branches=4, steps=steps, dendrite=dendrite,
               temporal=True, inhibition=True, adaptive_threshold=True,
               hard_spike=hard_spike, output_mode=bio_output,
               weight_rank=bio_rank)
    if ablation == "no_branching":
        cfg["branches"] = 1
    elif ablation == "no_temporal":
        cfg["temporal"] = False
    elif ablation == "no_inhibition":
        cfg["inhibition"] = False
    elif ablation == "fixed_threshold":
        cfg["adaptive_threshold"] = False
    elif ablation == "one_step":
        cfg["steps"] = 1
    return BioMLP(d_in, hidden, d_out, **cfg)


def save_scatter(path, x, y, title):
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 4))
        for cls in torch.unique(y).tolist():
            z = x[y == cls]
            plt.scatter(z[:, 0], z[:, 1], s=8, label=str(cls))
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()
    except Exception as exc:
        print(f"visualization skipped: {exc}")


def project_hidden(h):
    h = h - h.mean(0, keepdim=True)
    _, _, v = torch.pca_lowrank(h, q=min(2, h.shape[1]))
    return h @ v[:, :2]


def train_model(model, x_train, y_train, x_test, y_test, epochs, lr, device,
                batch_size=128):
    model.to(device)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # monotonic clock is used so wall-clock adjustments cannot corrupt the
    # reported benchmark time.
    start = time.monotonic()
    history = []
    for ep in range(epochs):
        model.train()
        order = torch.randperm(x_train.shape[0], device=device)
        running_loss = 0.0
        for start in range(0, x_train.shape[0], batch_size):
            ids = order[start:start + batch_size]
            logits = model(x_train[ids])
            loss = F.cross_entropy(logits, y_train[ids])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running_loss += loss.item() * ids.numel()
        model.eval()
        with torch.no_grad():
            tr_acc = (model(x_train).argmax(1) == y_train).float().mean().item()
            te_logits, hidden = model(x_test, return_hidden=True)
            te_loss = F.cross_entropy(te_logits, y_test).item()
            te_acc = (te_logits.argmax(1) == y_test).float().mean().item()
        history.append((running_loss / x_train.shape[0], te_loss, tr_acc, te_acc))
    elapsed = time.monotonic() - start
    with torch.no_grad():
        _, hidden = model(x_test, return_hidden=True)
    result = {
        "train_loss": history[-1][0], "test_loss": history[-1][1],
        "train_acc": history[-1][2], "test_acc": history[-1][3],
        "best_test_acc": max(row[3] for row in history),
        "history": [
            {"epoch": i + 1, "train_loss": row[0], "test_loss": row[1],
             "train_acc": row[2], "test_acc": row[3]}
            for i, row in enumerate(history)
        ],
        "parameters": sum(p.numel() for p in model.parameters()),
        "seconds": elapsed, "seconds_per_epoch": elapsed / epochs,
    }
    bio = getattr(model, "bio", None)
    if bio is not None:
        result["diagnostics"] = bio.last_diagnostics
        result["approx_flops_per_sample"] = bio.parameter_report()["approx_flops_per_step"] * bio.steps
    else:
        result["approx_flops_per_sample"] = sum(p.numel() for p in model.parameters()) * 2
    return result, hidden.detach().cpu()


def run_task(name, args, out_dir):
    if name == "mnist":
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader, Subset
        tf = transforms.ToTensor()
        train = datasets.MNIST(args.data_root, train=True, download=True, transform=tf)
        test = datasets.MNIST(args.data_root, train=False, download=True, transform=tf)
        if args.subset:
            train = Subset(train, torch.arange(args.subset))
        tl = DataLoader(train, batch_size=args.batch, shuffle=False)
        vl = DataLoader(test, batch_size=2048, shuffle=False)
        x_train = torch.cat([x.flatten(1) for x, _ in tl])
        y_train = torch.cat([y for _, y in tl])
        x_test = torch.cat([x.flatten(1) for x, _ in vl])
        y_test = torch.cat([y for _, y in vl])
        d_in, d_out, hidden = 784, 10, args.hidden
    else:
        x_train, y_train = synthetic(name, args.samples, args.seed)
        x_test, y_test = synthetic(name, args.test_samples, args.seed + 1)
        d_in, d_out, hidden = 2, 2, args.synthetic_hidden

    kinds = [("linear", None), ("relu", None), ("gelu", None), ("bio", None)]
    if args.ablation and name != "mnist":
        kinds = [("bio", a) for a in
                 ("no_branching", "no_temporal", "no_inhibition",
                  "fixed_threshold", "one_step")]
    rows = []
    for kind, ablation in kinds:
        torch.manual_seed(args.seed)
        model = make_model(kind, d_in, hidden, d_out, ablation,
                           args.steps, args.hard_spike, args.dendrite,
                           args.bio_rank, args.bio_output)
        result, h = train_model(model, x_train, y_train, x_test, y_test,
                                args.epochs, args.lr, args.device, args.batch)
        result.update({"task": name, "model": kind, "ablation": ablation})
        rows.append(result)
        print(f"{name:8s} {kind:6s} {str(ablation):18s} "
              f"test={result['test_acc']:.4f} best={result['best_test_acc']:.4f} "
              f"params={result['parameters']} time/ep={result['seconds_per_epoch']:.3f}s")
        if name != "mnist":
            save_scatter(os.path.join(out_dir, f"{name}_{kind}_{ablation or 'full'}_hidden.png"),
                         project_hidden(h), y_test, f"{name} / {kind} / {ablation or 'full'}")
        if kind == "bio" and ablation is None:
            bio = model.bio
            try:
                import matplotlib.pyplot as plt
                exc, inh = bio.effective_weights()
                w = (exc - inh).detach().cpu()
                plt.figure(figsize=(8, 5)); plt.imshow(w.flatten(0, 1), aspect="auto", cmap="coolwarm")
                plt.colorbar(); plt.xlabel("input feature"); plt.ylabel("neuron x branch")
                plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{name}_bio_dendritic_weights.png"), dpi=140); plt.close()
            except Exception as exc:
                print(f"weight heatmap skipped: {exc}")
    with open(os.path.join(out_dir, f"{name}_results.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["xor", "circles", "moons", "mnist", "all"], default="all")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--samples", type=int, default=1024)
    ap.add_argument("--test-samples", type=int, default=1024)
    ap.add_argument("--synthetic-hidden", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--subset", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dendrite", choices=["soft_threshold", "quadratic", "tanh"], default="soft_threshold")
    ap.add_argument("--bio-rank", type=int, default=0,
                    help="low-rank dendritic weights; 0 = full weights")
    ap.add_argument("--bio-output", choices=["mean", "final", "membrane"], default="mean")
    ap.add_argument("--hard-spike", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--data-root", default=os.path.join(ROOT, "data", "mnist"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out_dir = os.path.join(ROOT, "bio_results")
    os.makedirs(out_dir, exist_ok=True)
    tasks = ["xor", "circles", "moons", "mnist"] if args.task == "all" else [args.task]
    for task in tasks:
        if task == "mnist":
            args.lr = min(args.lr, 3e-3)
        run_task(task, args, out_dir)
    print(f"results written to {out_dir}")


if __name__ == "__main__":
    main()
