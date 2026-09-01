"""Parameter-matched ReLU/GELU MLP comparison against the current Bio results."""
import argparse, json, os, sys, time
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from benchmark_bio_simple_complex import make_data


class MLP(nn.Module):
    def __init__(self, d_in, hidden, classes, kind):
        super().__init__()
        act = nn.ReLU() if kind == "relu" else nn.GELU()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), act,
                                 nn.Linear(hidden, classes))

    def forward(self, x):
        return self.net(x.flatten(1))


def parameter_count(d_in, hidden, classes):
    return (d_in + 1) * hidden + (hidden + 1) * classes


def matched_hidden(d_in, classes, target):
    candidates = range(1, max(2, target // max(1, d_in + classes - 1)) + 3)
    return min(candidates, key=lambda h: abs(parameter_count(d_in, h, classes) - target))


def train(task, kind, hidden, args):
    xtr, xte, ytr, yte, classes = make_data(task, args.n, args.seed)
    model = MLP(xtr.shape[1], hidden, classes, kind).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_loader = DataLoader(TensorDataset(xtr, ytr), args.batch, shuffle=True)
    test_loader = DataLoader(TensorDataset(xte, yte), 1024)
    history = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train(); total = correct = loss_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(args.device), yb.to(args.device)
            logits = model(xb); loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            loss_sum += loss.item() * yb.numel()
            correct += (logits.argmax(1) == yb).sum().item(); total += yb.numel()
        model.eval(); good = ntest = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                logits = model(xb.to(args.device)); good += (logits.argmax(1) == yb.to(args.device)).sum().item(); ntest += yb.numel()
        history.append({"epoch": epoch, "train_acc": correct / total,
                        "test_acc": good / ntest, "train_loss": loss_sum / total})
    elapsed = time.perf_counter() - start
    return {"task": task, "model": kind, "hidden": hidden,
            "parameters": parameter_count(xtr.shape[1], hidden, classes),
            "seconds": elapsed, "best_test_acc": max(h["test_acc"] for h in history),
            "final_test_acc": history[-1]["test_acc"], "history": history}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="xor,circles,moons,checkerboard,parity8,noisy_moons100,noisy_spiral100")
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=128); ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bio-reference", default="bio_results/bio_simple_complex_seed0.json")
    ap.add_argument("--result", required=True)
    args = ap.parse_args(); torch.manual_seed(args.seed)
    bio = {r["task"]: r for r in json.load(open(args.bio_reference, encoding="utf-8"))}
    rows = []
    for task in args.tasks.split(","):
        # Bio's parameter count depends on input dimension and class count.
        xtr, _, _, _, classes = make_data(task, args.n, args.seed)
        target = bio[task]["parameters"]
        hidden = matched_hidden(xtr.shape[1], classes, target)
        print(f"[{task}] target_bio={target} matched_hidden={hidden} mlp_params={parameter_count(xtr.shape[1],hidden,classes)}", flush=True)
        for kind in ("relu", "gelu"):
            torch.manual_seed(args.seed)
            rows.append(train(task, kind, hidden, args))
            r = rows[-1]; print(f"  {kind} best={r['best_test_acc']:.4f} final={r['final_test_acc']:.4f} time={r['seconds']:.1f}s", flush=True)
    out = {"config": vars(args), "bio_reference": bio, "results": rows}
    os.makedirs(os.path.dirname(args.result) or ".", exist_ok=True)
    with open(args.result, "w", encoding="utf-8") as f: json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__": main()
