"""Run the best Minimal Local NF (A-1step) on the existing hard synthetic tasks.

This is intentionally independent from the historical benchmark files so that
their results are not overwritten.  The model keeps the winning 256-node,
16x16, one-step local field and only changes the input projection dimension.
"""
import argparse, json, os, time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.datasets import make_moons


def make_data(name, n, seed):
    rng = np.random.RandomState(seed)
    if name == "spiral3":
        per = n // 3; xs = []; ys = []
        for c in range(3):
            r = np.linspace(.05, 1, per)
            th = np.linspace(c * 2 * np.pi / 3, (c + 1) * 2 * np.pi / 3, per) + rng.randn(per) * .18
            xs.append(np.c_[r * np.cos(th), r * np.sin(th)]); ys.extend([c] * per)
        x = np.concatenate(xs); y = np.array(ys, "int64")
    elif name == "checkerboard":
        x = rng.uniform(-1, 1, (n, 2)).astype("float32")
        bins = np.floor((x + 1) * 4).astype("int64")
        y = ((bins[:, 0] + bins[:, 1]) % 2).astype("int64")
        x += rng.randn(n, 2).astype("float32") * .025
    elif name == "parity8":
        x = rng.choice([-1., 1.], size=(n, 8)).astype("float32")
        y = ((x > 0).sum(1) % 2).astype("int64")
    elif name == "noisy_moons100":
        sig, y = make_moons(n_samples=n, noise=.20, random_state=seed)
        x = np.concatenate([sig.astype("float32"), rng.randn(n, 98).astype("float32")], 1)
    elif name == "noisy_spiral100":
        base_rng = np.random.RandomState(seed); per = n // 3; xs = []; ys = []
        for c in range(3):
            r = np.linspace(.05, 1, per)
            th = np.linspace(c * 2 * np.pi / 3, (c + 1) * 2 * np.pi / 3, per) + base_rng.randn(per) * .18
            xs.append(np.c_[r * np.cos(th), r * np.sin(th)]); ys.extend([c] * per)
        sig = np.concatenate(xs).astype("float32")
        x = np.concatenate([sig, base_rng.randn(len(sig), 98).astype("float32")], 1)
        y = np.array(ys, "int64")
    else:
        raise ValueError(name)
    order = rng.permutation(len(y)); split = int(.8 * len(y)); tr, te = order[:split], order[split:]
    mu = x[tr].mean(0, keepdims=True); sd = x[tr].std(0, keepdims=True) + 1e-6
    return (torch.from_numpy(((x[tr] - mu) / sd).astype("float32")),
            torch.from_numpy(((x[te] - mu) / sd).astype("float32")),
            torch.from_numpy(y[tr].astype("int64")), torch.from_numpy(y[te].astype("int64")), int(y.max() + 1))


class MinimalField(nn.Module):
    def __init__(self, size=16, steps=1):
        super().__init__()
        self.steps = steps
        self.decay_logit = nn.Parameter(torch.tensor(1.3863))
        self.kernel = nn.Parameter(torch.tensor([[0., .2, 0.], [.2, 0., .2], [0., .2, 0.]]).view(1, 1, 3, 3))

    def forward(self, x):
        v = x
        for _ in range(self.steps):
            signal = torch.tanh(v)
            v = torch.sigmoid(self.decay_logit) * v + F.conv2d(signal, self.kernel, padding=1)
        return v


class MinimalLocalNF(nn.Module):
    def __init__(self, d_in, classes, steps=1):
        super().__init__()
        self.up = nn.Linear(d_in, 256)
        self.field = MinimalField(steps=steps)
        self.out = nn.Linear(256, classes)

    def forward(self, x):
        v = self.up(x).view(x.shape[0], 1, 16, 16)
        return self.out(self.field(v).flatten(1))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); good = total = 0
    for x, y in loader:
        z = model(x.to(device)); good += (z.argmax(1) == y.to(device)).sum().item(); total += y.numel()
    return good / total


def run(task, args, device, steps):
    xtr, xte, ytr, yte, classes = make_data(task, args.n, args.seed)
    tr = TensorDataset(xtr, ytr); te = TensorDataset(xte, yte)
    train = DataLoader(tr, args.batch, shuffle=True); test = DataLoader(te, 1024)
    model = MinimalLocalNF(xtr.shape[1], classes, steps=steps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    hist = []; start = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        model.train(); correct = total = loss_sum = 0.
        for xb, yb in train:
            xb, yb = xb.to(device), yb.to(device)
            z = model(xb); loss = F.cross_entropy(z, yb)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step()
            loss_sum += loss.item() * yb.numel(); correct += (z.argmax(1) == yb).sum().item(); total += yb.numel()
        acc = evaluate(model, test, device)
        hist.append({"epoch": ep, "train_loss": loss_sum / total, "train_acc": correct / total, "test_acc": acc})
        print(f"[{task}] ep={ep:03d} train={correct/total:.4f} test={acc:.4f}", flush=True)
    return {"task": task, "model": f"minimal_local_nf_a_{steps}step", "steps": steps,
            "parameters": sum(p.numel() for p in model.parameters()),
            "best_test_acc": max(r["test_acc"] for r in hist), "final_test_acc": hist[-1]["test_acc"],
            "seconds": time.perf_counter() - start, "history": hist}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="spiral3,checkerboard,parity8,noisy_moons100,noisy_spiral100")
    p.add_argument("--n", type=int, default=6000); p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=128); p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--steps", default="1", help="comma-separated field step counts, e.g. 2,3")
    p.add_argument("--result", default="hard_task_results/minimal_local_nf_a1_seed0.json")
    a = p.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); device = torch.device(a.device)
    results = [run(t, a, device, int(s)) for s in a.steps.split(",") for t in a.tasks.split(",")]
    os.makedirs(os.path.dirname(a.result) or ".", exist_ok=True)
    with open(a.result, "w", encoding="utf-8") as f: json.dump({"config": vars(a), "results": results}, f, indent=2)
    print("SUMMARY")
    for r in results: print(f"{r['task']:18s} best={r['best_test_acc']:.4f} final={r['final_test_acc']:.4f} time={r['seconds']:.1f}s")


if __name__ == "__main__": main()
