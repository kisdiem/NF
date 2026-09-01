"""Independent-train/test multi-seed BioNeuron ablation on Noisy Spiral100."""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from models.bio_neuron import BioMLP


def spiral100(n, seed):
    rng = np.random.RandomState(seed); per = n // 3; xs = []; ys = []
    for c in range(3):
        r = np.linspace(.05, 1, per)
        th = np.linspace(c * 2 * np.pi / 3, (c + 1) * 2 * np.pi / 3, per) + rng.randn(per) * .18
        xs.append(np.c_[r * np.cos(th), r * np.sin(th)]); ys.extend([c] * per)
    sig = np.concatenate(xs).astype("float32")
    x = np.concatenate([sig, rng.randn(len(sig), 98).astype("float32")], 1)
    return x, np.array(ys, "int64")


def independent_split(n, seed):
    xtr, ytr = spiral100(n, seed)
    xte, yte = spiral100(n, seed + 10000)
    mu = xtr.mean(0, keepdims=True); sd = xtr.std(0, keepdims=True) + 1e-6
    norm = lambda x: torch.from_numpy(((x - mu) / sd).astype("float32"))
    return norm(xtr), norm(xte), torch.from_numpy(ytr), torch.from_numpy(yte)


def make_model(variant):
    cfg = dict(branches=4, steps=3, dendrite="soft_threshold", temporal=True,
               inhibition=True, adaptive_threshold=True, output_mode="mean")
    if variant == "no_branching": cfg["branches"] = 1
    elif variant == "no_temporal": cfg["temporal"] = False
    elif variant == "no_inhibition": cfg["inhibition"] = False
    elif variant == "fixed_threshold": cfg["adaptive_threshold"] = False
    elif variant == "no_membrane_persistence": cfg["membrane_decay"] = 0.0
    elif variant == "one_step": cfg["steps"] = 1
    elif variant != "full": raise ValueError(variant)
    return BioMLP(100, 64, 3, **cfg)


def run(variant, seed, args):
    xtr, xte, ytr, yte = independent_split(args.n, seed)
    torch.manual_seed(seed); model = make_model(variant).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train = DataLoader(TensorDataset(xtr, ytr), args.batch, shuffle=True)
    test = DataLoader(TensorDataset(xte, yte), 1024)
    history = []; t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train(); total = correct = loss_sum = 0.0
        for xb, yb in train:
            xb, yb = xb.to(args.device), yb.to(args.device)
            logits = model(xb); loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            loss_sum += loss.item() * yb.numel(); correct += (logits.argmax(1) == yb).sum().item(); total += yb.numel()
        model.eval(); good = ntest = 0
        with torch.no_grad():
            for xb, yb in test:
                logits = model(xb.to(args.device)); good += (logits.argmax(1) == yb.to(args.device)).sum().item(); ntest += yb.numel()
        history.append({"epoch": epoch, "train_loss": loss_sum / total,
                        "train_acc": correct / total, "test_acc": good / ntest})
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"[{variant} seed={seed}] ep={epoch:03d} train={correct/total:.4f} test={good/ntest:.4f}", flush=True)
    seconds = time.perf_counter() - t0
    return {"model": "BioMLP", "variant": variant, "seed": seed,
            "train_samples": args.n, "test_samples": args.n,
            "independent_test_seed": seed + 10000,
            "parameters": sum(p.numel() for p in model.parameters()),
            "seconds": seconds, "seconds_per_epoch": seconds / args.epochs,
            "best_test_acc": max(h["test_acc"] for h in history),
            "final_test_acc": history[-1]["test_acc"], "history": history,
            "diagnostics": model.bio.last_diagnostics,
            "approx_flops_per_sample": model.bio.parameter_report()["approx_flops_per_step"] * model.bio.steps}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--variants", default="full,no_branching,no_temporal,no_inhibition,fixed_threshold,no_membrane_persistence,one_step")
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=128); ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--result", required=True)
    args = ap.parse_args(); os.makedirs(os.path.dirname(args.result) or ".", exist_ok=True)
    rows = []; seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    for seed in seeds:
        for variant in args.variants.split(","):
            row = run(variant, seed, args); rows.append(row)
            with open(args.result, "w", encoding="utf-8") as f:
                json.dump({"config": vars(args), "results": rows}, f, indent=2, ensure_ascii=False)
    print("SUMMARY")
    for variant in args.variants.split(","):
        rr = [r for r in rows if r["variant"] == variant]
        print(f"{variant:25s} mean_best={np.mean([r['best_test_acc'] for r in rr]):.4f} "
              f"std={np.std([r['best_test_acc'] for r in rr]):.4f} "
              f"mean_final={np.mean([r['final_test_acc'] for r in rr]):.4f}")


if __name__ == "__main__": main()
