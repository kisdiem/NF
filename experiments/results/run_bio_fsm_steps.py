"""Compare Bio step counts and dynamics on independent FSM Sequence splits."""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from models.bio_neuron import BioMLP
from experiments.results.run_bio_easy_old_logic_suite import data


def make_model(variant):
    if variant == "bio-eazy-3step":
        cfg = dict(steps=3, temporal=False, membrane_decay=0.0)
    elif variant == "bio-eazy-5step":
        cfg = dict(steps=5, temporal=False, membrane_decay=0.0)
    elif variant == "bio-old-5step":
        cfg = dict(steps=5, temporal=True, membrane_decay=0.8)
    else:
        raise ValueError(variant)
    return BioMLP(36, 64, 3, branches=4, dendrite="soft_threshold",
                  inhibition=True, adaptive_threshold=True, output_mode="mean", **cfg)


def run(variant, seed, args):
    xtr, xte, ytr, yte, _ = data("fsm_sequence", args.n, seed)
    torch.manual_seed(seed); model = make_model(variant).to(args.device)
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
            print(f"[{variant} seed={seed}] ep={epoch:03d} train={correct/total:.4f} test={good/ntest:.4f}", flush=True)
    seconds = time.perf_counter() - start
    return {"task": "fsm_sequence", "variant": variant, "seed": seed,
            "independent_test_seed": seed + 10000, "parameters": sum(p.numel() for p in model.parameters()),
            "seconds": seconds, "seconds_per_epoch": seconds / args.epochs,
            "best_test_acc": max(h["test_acc"] for h in history), "final_test_acc": history[-1]["test_acc"],
            "history": history, "diagnostics": model.bio.last_diagnostics}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n", type=int, default=6000); ap.add_argument("--batch", type=int, default=128); ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--result", required=True)
    args = ap.parse_args(); os.makedirs(os.path.dirname(args.result) or ".", exist_ok=True); rows=[]
    for seed in [int(x) for x in args.seeds.split(",") if x.strip()]:
        for variant in ("bio-eazy-3step", "bio-eazy-5step", "bio-old-5step"):
            row=run(variant,seed,args); rows.append(row)
            with open(args.result,"w",encoding="utf-8") as f: json.dump({"config":vars(args),"results":rows},f,indent=2,ensure_ascii=False)
    print("SUMMARY")
    for variant in ("bio-eazy-3step", "bio-eazy-5step", "bio-old-5step"):
        rr=[r for r in rows if r["variant"]==variant]
        print(f"{variant:18s} mean_best={np.mean([r['best_test_acc'] for r in rr]):.4f} std={np.std([r['best_test_acc'] for r in rr]):.4f} mean_final={np.mean([r['final_test_acc'] for r in rr]):.4f}")

if __name__=="__main__": main()
