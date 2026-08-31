"""Temporal sanity checks: sMNIST + Adding Problem.

Tests whether the NF field's delay/duration dynamics help on SEQUENCE tasks.
Sequence time is mapped onto the field's own K_t steps (Option B): each
sequence timestep is injected as a seed-field, and the field propagates across
steps. Compared against:
  - LSTM        (stateful baseline, the bar to beat)
  - PerStepReLU (same per-step structure as NF but no cross-step state,
                 isolates "memory" from "nonlinearity")

Run:
  python train_seq.py --task smnist --model nf
  python train_seq.py --task adding --model nf
"""
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

from nf_field import DiscreteNeuralField

SEQ_FIELD = dict(H=32, W=32, K_s=16, L_max=8, D_max=3, tau=1.0,
                 inject_scale=2.0, surr_scale=1.0, read_mode="potential")


class SeqFieldModel(nn.Module):
    """Sequence fed through the field's own time."""

    def __init__(self, feat, d, out, T):
        super().__init__()
        self.W_in = nn.Linear(feat, d)
        self.field = DiscreteNeuralField(d, K_t=T, R=min(12, T), **SEQ_FIELD)
        self.W_out = nn.Linear(d, out)
        nn.init.uniform_(self.W_out.weight, -0.05, 0.05)
        nn.init.uniform_(self.W_out.bias, -0.05, 0.05)

    def forward(self, x):               # x: (B, T, feat)
        h = self.W_in(x)                # (B, T, d)
        v = self.field.forward_seq(h)   # (B, d)
        return self.W_out(v)


class LSTM(nn.Module):
    def __init__(self, feat, d, out, num_layers=1):
        super().__init__()
        self.rnn = nn.LSTM(feat, d, num_layers, batch_first=True)
        self.W_out = nn.Linear(d, out)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.W_out(out[:, -1])


class PerStepReLU(nn.Module):
    """Same per-step structure as SeqField but stateless (mean-pool over steps)."""

    def __init__(self, feat, d, out):
        super().__init__()
        self.W_in = nn.Linear(feat, d)
        self.W_out = nn.Linear(d, out)

    def forward(self, x):
        h = F.relu(self.W_in(x))
        return self.W_out(h.mean(dim=1))


# ---------------------------------------------------------------- data ----- #
def load_smnist(root, subset, batch):
    tf = transforms.ToTensor()
    train_ds = datasets.MNIST(root=root, train=True, download=True, transform=tf)
    test_ds = datasets.MNIST(root=root, train=False, download=True, transform=tf)
    if subset:
        train_ds = Subset(train_ds, torch.arange(subset))
    tr = DataLoader(train_ds, batch_size=batch, shuffle=True)
    te = DataLoader(test_ds, batch_size=1024, shuffle=False)
    return tr, te


def make_adding(T, n_train, n_test, seed=0):
    """Classic Adding Problem: [value, flag]; two flagged positions; target =
    sum of the two values at flagged positions (Hochreiter & Schmidhuber 1997)."""
    g = torch.Generator().manual_seed(seed)
    def gen(n):
        x = torch.rand(n, T, 2, generator=g)      # value ~ U(0,1), flag = 0
        x[:, :, 1] = 0.0
        pos = torch.stack([torch.randperm(T)[:2] for _ in range(n)])  # two marks
        x[torch.arange(n).unsqueeze(1), pos, 1] = 1.0                # set flags only
        y = x[torch.arange(n).unsqueeze(1), pos][:, :, 0].sum(dim=1)  # sum values at flags
        return x, y
    xt, yt = gen(n_train)
    xe, ye = gen(n_test)
    return (DataLoader(TensorDataset(xt, yt), batch_size=128, shuffle=True),
            DataLoader(TensorDataset(xe, ye), batch_size=1024))


# ------------------------------------------------------------- train ----- #
def build(args):
    if args.model == "nf":
        return SeqFieldModel(args.feat, 64, args.out, args.T)
    if args.model == "lstm":
        return LSTM(args.feat, 64, args.out)
    if args.model == "relu":
        return PerStepReLU(args.feat, 64, args.out)
    raise ValueError(args.model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["smnist", "adding"], required=True)
    ap.add_argument("--model", choices=["nf", "lstm", "relu"], required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--subset", type=int, default=20000)
    ap.add_argument("--T", type=int, default=20, help="sequence length (adding)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", default="data/mnist")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = args.device
    is_add = args.task == "adding"
    args.feat = 2 if is_add else 28
    args.out = 1 if is_add else 10
    args.T = args.T if is_add else 28

    model = build(args).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if is_add:
        train_loader, test_loader = make_adding(args.T, args.subset, 2000)
    else:
        train_loader, test_loader = load_smnist(args.data_root, args.subset, args.batch)

    print(f"[{args.task}:{args.model}] T={args.T} feat={args.feat} "
          f"subset={args.subset} epochs={args.epochs} lr={args.lr}")
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        run_loss = run_n = 0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            if not is_add:
                x = x.squeeze(1)                      # (B,1,28,28) -> (B,28,28)
            opt.zero_grad()
            out = model(x).squeeze(-1)
            loss = F.mse_loss(out, y) if is_add else F.cross_entropy(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += loss.item() * y.numel()
            run_n += y.numel()
        # eval
        model.eval()
        correct = n = 0
        tot = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(dev), y.to(dev)
                if not is_add:
                    x = x.squeeze(1)                  # (B,1,28,28) -> (B,28,28)
                out = model(x).squeeze(-1)
                if is_add:
                    tot += F.mse_loss(out, y, reduction="sum").item()
                    correct += ((out - y).abs() < 0.04).sum().item()
                else:
                    correct += (out.argmax(1) == y).sum().item()
                n += y.numel()
        metric = f"acc={correct/n:.4f}" if not is_add else f"mse={tot/n:.5f} acc04={correct/n:.4f}"
        print(f"[{args.task}:{args.model}] ep{ep+1} loss={run_loss/run_n:.4f} {metric} "
              f"elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
