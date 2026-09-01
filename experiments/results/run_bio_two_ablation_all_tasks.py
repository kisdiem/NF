"""Compare two Bio simplifications on all simple-to-complex synthetic tasks."""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import make_moons, make_circles
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from models.bio_neuron import BioMLP


def raw_data(name, n, seed):
    rng = np.random.RandomState(seed)
    if name == "xor":
        x = rng.choice([-1., 1.], (n, 2)).astype("float32")
        x += rng.randn(n, 2).astype("float32") * .08
        y = ((x[:, 0] * x[:, 1]) < 0).astype("int64")
    elif name == "circles":
        x, y = make_circles(n, noise=.12, factor=.5, random_state=seed)
        x, y = x.astype("float32"), y.astype("int64")
    elif name == "moons":
        x, y = make_moons(n, noise=.18, random_state=seed)
        x, y = x.astype("float32"), y.astype("int64")
    elif name == "checkerboard":
        x = rng.uniform(-1, 1, (n, 2)).astype("float32")
        bins = np.floor((x + 1) * 4).astype("int64")
        y = ((bins[:, 0] + bins[:, 1]) % 2).astype("int64")
        x += rng.randn(n, 2).astype("float32") * .025
    elif name == "parity8":
        x = rng.choice([-1., 1.], (n, 8)).astype("float32")
        y = ((x > 0).sum(1) % 2).astype("int64")
    elif name == "noisy_moons100":
        sig, y = make_moons(n, noise=.20, random_state=seed)
        x = np.concatenate([sig.astype("float32"), rng.randn(n, 98).astype("float32")], 1)
    elif name == "noisy_spiral100":
        base = np.random.RandomState(seed); per = n // 3; xs = []; ys = []
        for c in range(3):
            r = np.linspace(.05, 1, per)
            th = np.linspace(c * 2 * np.pi / 3, (c + 1) * 2 * np.pi / 3, per) + base.randn(per) * .18
            xs.append(np.c_[r * np.cos(th), r * np.sin(th)]); ys.extend([c] * per)
        sig = np.concatenate(xs).astype("float32")
        x = np.concatenate([sig, base.randn(len(sig), 98).astype("float32")], 1)
        y = np.array(ys, "int64")
    else:
        raise ValueError(name)
    return x, y


def independent_data(name, n, seed):
    xtr, ytr = raw_data(name, n, seed)
    xte, yte = raw_data(name, n, seed + 10000)
    mu = xtr.mean(0, keepdims=True); sd = xtr.std(0, keepdims=True) + 1e-6
    norm = lambda x: torch.from_numpy(((x - mu) / sd).astype("float32"))
    return norm(xtr), norm(xte), torch.from_numpy(ytr), torch.from_numpy(yte), int(ytr.max() + 1)


def make_model(variant, d_in, classes):
    cfg = dict(branches=4, steps=3, dendrite="soft_threshold", temporal=True,
               inhibition=True, adaptive_threshold=True, output_mode="mean")
    if variant == "no_temporal_no_membrane":
        cfg["temporal"] = False; cfg["membrane_decay"] = 0.0
    elif variant == "one_step_temporal_membrane":
        cfg["steps"] = 1
    else:
        raise ValueError(variant)
    return BioMLP(d_in, 64, classes, **cfg)


def run(task, variant, seed, args):
    xtr, xte, ytr, yte, classes = independent_data(task, args.n, seed)
    torch.manual_seed(seed); model = make_model(variant, xtr.shape[1], classes).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train = DataLoader(TensorDataset(xtr, ytr), args.batch, shuffle=True)
    test = DataLoader(TensorDataset(xte, yte), 1024)
    history = []; start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train(); total = correct = loss_sum = 0.0
        for xb, yb in train:
            xb, yb = xb.to(args.device), yb.to(args.device); logits = model(xb); loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            loss_sum += loss.item() * yb.numel(); correct += (logits.argmax(1) == yb).sum().item(); total += yb.numel()
        model.eval(); good = ntest = 0
        with torch.no_grad():
            for xb, yb in test:
                logits = model(xb.to(args.device)); good += (logits.argmax(1) == yb.to(args.device)).sum().item(); ntest += yb.numel()
        history.append({"epoch": epoch, "train_loss": loss_sum / total, "train_acc": correct / total, "test_acc": good / ntest})
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"[{task} {variant} seed={seed}] ep={epoch:03d} train={correct/total:.4f} test={good/ntest:.4f}", flush=True)
    seconds = time.perf_counter() - start
    return {"task": task, "variant": variant, "seed": seed, "independent_test_seed": seed + 10000,
            "parameters": sum(p.numel() for p in model.parameters()), "seconds": seconds,
            "seconds_per_epoch": seconds / args.epochs, "best_test_acc": max(h["test_acc"] for h in history),
            "final_test_acc": history[-1]["test_acc"], "history": history,
            "diagnostics": model.bio.last_diagnostics}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tasks", default="xor,circles,moons,checkerboard,parity8,noisy_moons100,noisy_spiral100")
    ap.add_argument("--variants", default="no_temporal_no_membrane,one_step_temporal_membrane")
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=128); ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--result", required=True)
    args = ap.parse_args(); os.makedirs(os.path.dirname(args.result) or ".", exist_ok=True)
    rows = []
    for seed in [int(x) for x in args.seeds.split(",") if x.strip()]:
        for task in args.tasks.split(","):
            for variant in args.variants.split(","):
                row = run(task, variant, seed, args); rows.append(row)
                with open(args.result, "w", encoding="utf-8") as f: json.dump({"config": vars(args), "results": rows}, f, indent=2, ensure_ascii=False)
    print("SUMMARY")
    for task in args.tasks.split(","):
        for variant in args.variants.split(","):
            rr = [r for r in rows if r["task"] == task and r["variant"] == variant]
            print(f"{task:18s} {variant:28s} mean_best={np.mean([r['best_test_acc'] for r in rr]):.4f} std={np.std([r['best_test_acc'] for r in rr]):.4f}")


if __name__ == "__main__": main()
