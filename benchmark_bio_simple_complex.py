"""Run the current BioNeuron on the same simple/complex synthetic tasks as Local NF."""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons, make_circles
from models.bio_neuron import BioMLP


def make_data(name, n, seed):
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
        base = np.random.RandomState(seed); per = n // 3; xs=[]; ys=[]
        for c in range(3):
            r = np.linspace(.05, 1, per)
            th = np.linspace(c*2*np.pi/3, (c+1)*2*np.pi/3, per) + base.randn(per)*.18
            xs.append(np.c_[r*np.cos(th), r*np.sin(th)]); ys.extend([c]*per)
        sig = np.concatenate(xs).astype("float32")
        x = np.concatenate([sig, base.randn(len(sig), 98).astype("float32")], 1)
        y = np.array(ys, "int64")
    else: raise ValueError(name)
    order = rng.permutation(len(y)); split = int(.8*len(y)); tr, te = order[:split], order[split:]
    mu, sd = x[tr].mean(0, keepdims=True), x[tr].std(0, keepdims=True)+1e-6
    norm = lambda a: torch.from_numpy(((a-mu)/sd).astype("float32"))
    return norm(x[tr]), norm(x[te]), torch.from_numpy(y[tr]), torch.from_numpy(y[te]), int(y.max()+1)


def run(name, args):
    xtr, xte, ytr, yte, classes = make_data(name, args.n, args.seed)
    model = BioMLP(xtr.shape[1], args.hidden, classes, branches=4, steps=args.steps,
                   dendrite=args.dendrite, temporal=True, inhibition=True,
                   adaptive_threshold=True, output_mode="mean").to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train = DataLoader(TensorDataset(xtr,ytr), args.batch, shuffle=True)
    test = DataLoader(TensorDataset(xte,yte), 1024)
    hist=[]; t0=time.perf_counter()
    for ep in range(1,args.epochs+1):
        model.train(); loss_sum=correct=total=0
        for xb,yb in train:
            xb,yb=xb.to(args.device),yb.to(args.device); z=model(xb); loss=F.cross_entropy(z,yb)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
            loss_sum += loss.item()*yb.numel(); correct += (z.argmax(1)==yb).sum().item(); total += yb.numel()
        model.eval(); good=0; nte=0
        with torch.no_grad():
            for xb,yb in test:
                z=model(xb.to(args.device)); good += (z.argmax(1)==yb.to(args.device)).sum().item(); nte += yb.numel()
        hist.append({"epoch":ep,"train_acc":correct/total,"test_acc":good/nte,"train_loss":loss_sum/total})
        print(f"[{name}] ep={ep:03d} train={correct/total:.4f} test={good/nte:.4f}", flush=True)
    elapsed=time.perf_counter()-t0
    return {"task":name,"model":"BioMLP","config":vars(args),"parameters":sum(p.numel() for p in model.parameters()),
            "seconds":elapsed,"seconds_per_epoch":elapsed/args.epochs,"best_test_acc":max(h["test_acc"] for h in hist),
            "final_test_acc":hist[-1]["test_acc"],"history":hist,"diagnostics":model.bio.last_diagnostics,
            "approx_flops_per_sample":model.bio.parameter_report()["approx_flops_per_step"]*args.steps}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--tasks",default="xor,circles,moons,checkerboard,parity8,noisy_moons100,noisy_spiral100")
    ap.add_argument("--epochs",type=int,default=100); ap.add_argument("--n",type=int,default=6000); ap.add_argument("--batch",type=int,default=128)
    ap.add_argument("--hidden",type=int,default=64); ap.add_argument("--steps",type=int,default=3); ap.add_argument("--lr",type=float,default=3e-3)
    ap.add_argument("--seed",type=int,default=0); ap.add_argument("--dendrite",default="soft_threshold",choices=["soft_threshold","quadratic","tanh"])
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--result",default="bio_results/bio_simple_complex_seed0.json")
    args=ap.parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed); os.makedirs(os.path.dirname(args.result) or ".",exist_ok=True)
    rows=[]
    for task in args.tasks.split(","):
        torch.manual_seed(args.seed); rows.append(run(task,args))
    with open(args.result,"w",encoding="utf-8") as f: json.dump(rows,f,indent=2,ensure_ascii=False)
    print("SUMMARY"); [print(f"{r['task']:18s} best={r['best_test_acc']:.4f} final={r['final_test_acc']:.4f} time={r['seconds']:.1f}s") for r in rows]

if __name__=="__main__": main()
