"""Minimal Local NF experiment.

A: membrane potential only.
B: membrane potential + per-node threshold.
Both use 256 scalar nodes (16x16), a shared learnable local kernel, and no
inhibition/strength/refractory attributes.

Examples:
  python train_simple_field_ab.py --epochs 30 --variants A_1step
  python train_simple_field_ab.py --epochs 20 --variants A_0step_linear_control,A_1step,A_2step,A_3step
  python train_simple_field_ab.py --epochs 20 --variants B_1step,B_3step
"""
import argparse
import json
import time

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MinimalField(nn.Module):
    def __init__(self, threshold=False, steps=2, size=16, circular=False):
        super().__init__()
        self.threshold = threshold
        self.steps = steps
        self.size = size
        self.circular = circular
        self.decay_logit = nn.Parameter(torch.tensor(1.3863))
        self.kernel = nn.Parameter(
            torch.tensor([[0.0, 0.2, 0.0], [0.2, 0.0, 0.2], [0.0, 0.2, 0.0]])
            .view(1, 1, 3, 3)
        )
        if threshold:
            self.theta = nn.Parameter(torch.zeros(1, 1, size, size))

    def forward(self, x):
        v = x
        decay = torch.sigmoid(self.decay_logit)
        for _ in range(self.steps):
            if self.threshold:
                signal = torch.sigmoid((v - self.theta) / 0.5)
            else:
                signal = torch.tanh(v)
            if self.circular:
                signal = F.pad(signal, (1, 1, 1, 1), mode="circular")
                incoming = F.conv2d(signal, self.kernel, padding=0)
            else:
                incoming = F.conv2d(signal, self.kernel, padding=1)
            v = decay * v + incoming
        return v


class FieldMLP(nn.Module):
    def __init__(self, threshold, steps=2, circular=False):
        super().__init__()
        self.up = nn.Linear(784, 256)
        self.field = MinimalField(threshold, steps, circular=circular)
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        v = self.up(x.flatten(1)).view(-1, 1, 16, 16)
        return self.out(self.field(v).flatten(1))


def evaluate(model, loader, device):
    model.eval()
    good = total = 0
    with torch.no_grad():
        for x, y in loader:
            z = model(x.to(device))
            good += (z.argmax(1) == y.to(device)).sum().item()
            total += y.numel()
    return good / total


def run(model, train, test, device, epochs, lr):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    hist = []
    start = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        for x, y in train:
            loss = F.cross_entropy(model(x.to(device)), y.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_sum += loss.item() * y.numel()
            n += y.numel()
        row = {
            "epoch": ep,
            "train_loss": loss_sum / n,
            "test_acc": evaluate(model, test, device),
        }
        hist.append(row)
        print(row, flush=True)
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_test_acc": max(x["test_acc"] for x in hist),
        "final_test_acc": hist[-1]["test_acc"],
        "history": hist,
        "seconds": time.perf_counter() - start,
    }


def build_variant(name):
    specs = {
        "A_0step_linear_control": dict(threshold=False, steps=0, circular=False),
        "A_1step": dict(threshold=False, steps=1, circular=False),
        "A_2step": dict(threshold=False, steps=2, circular=False),
        "A_3step": dict(threshold=False, steps=3, circular=False),
        "A_1step_circular": dict(threshold=False, steps=1, circular=True),
        "A_3step_circular": dict(threshold=False, steps=3, circular=True),
        "B_1step": dict(threshold=True, steps=1, circular=False),
        "B_2step": dict(threshold=True, steps=2, circular=False),
        "B_3step": dict(threshold=True, steps=3, circular=False),
    }
    if name not in specs:
        raise ValueError(
            f"unknown variant {name!r}; choose from: {', '.join(specs)}"
        )
    return FieldMLP(**specs[name])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--data-root", default="data/mnist")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--subset", type=int, default=0)
    p.add_argument(
        "--variants",
        default="A_1step",
        help="comma-separated variants, e.g. A_0step_linear_control,A_1step,A_2step,A_3step",
    )
    p.add_argument("--result", default="simple_field_results.json")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    tf = transforms.ToTensor()
    tr = datasets.MNIST(a.data_root, True, download=True, transform=tf)
    te = datasets.MNIST(a.data_root, False, download=True, transform=tf)
    if a.subset:
        tr = torch.utils.data.Subset(tr, range(a.subset))

    train = DataLoader(tr, a.batch, shuffle=True)
    test = DataLoader(te, 1024)
    results = {"config": vars(a), "variants": {}}

    for name in [x.strip() for x in a.variants.split(",") if x.strip()]:
        # Reset seed before model creation so each variant is reproducible.
        torch.manual_seed(a.seed)
        model = build_variant(name)
        print("START", name, flush=True)
        results["variants"][name] = run(model, train, test, device, a.epochs, a.lr)

    print(json.dumps(results, indent=2))
    with open(a.result, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
