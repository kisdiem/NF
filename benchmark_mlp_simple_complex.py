"""Matched simple/complex benchmark for Linear, ReLU-MLP and GELU-MLP."""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons, make_circles

def make_data(name,n,seed):
    rng=np.random.RandomState(seed)
    if name=="xor":
        x=rng.choice([-1.,1.],(n,2)).astype("float32"); x+=rng.randn(n,2).astype("float32")*.08; y=((x[:,0]*x[:,1])<0).astype("int64")
    elif name=="circles": x,y=make_circles(n,noise=.12,factor=.5,random_state=seed); x=x.astype("float32"); y=y.astype("int64")
    elif name=="moons": x,y=make_moons(n,noise=.18,random_state=seed); x=x.astype("float32"); y=y.astype("int64")
    elif name=="checkerboard":
        x=rng.uniform(-1,1,(n,2)).astype("float32"); bins=np.floor((x+1)*4).astype("int64"); y=((bins[:,0]+bins[:,1])%2).astype("int64"); x+=rng.randn(n,2).astype("float32")*.025
    elif name=="parity8": x=rng.choice([-1.,1.],(n,8)).astype("float32"); y=((x>0).sum(1)%2).astype("int64")
    elif name=="noisy_moons100":
        sig,y=make_moons(n,noise=.20,random_state=seed); x=np.concatenate([sig.astype("float32"),rng.randn(n,98).astype("float32")],1)
    elif name=="noisy_spiral100":
        base=np.random.RandomState(seed); per=n//3; xs=[]; ys=[]
        for c in range(3):
            r=np.linspace(.05,1,per); th=np.linspace(c*2*np.pi/3,(c+1)*2*np.pi/3,per)+base.randn(per)*.18; xs.append(np.c_[r*np.cos(th),r*np.sin(th)]); ys.extend([c]*per)
        sig=np.concatenate(xs).astype("float32"); x=np.concatenate([sig,base.randn(len(sig),98).astype("float32")],1); y=np.array(ys,"int64")
    else: raise ValueError(name)
    order=rng.permutation(len(y)); split=int(.8*len(y)); tr,te=order[:split],order[split:]; mu=x[tr].mean(0,keepdims=True); sd=x[tr].std(0,keepdims=True)+1e-6
    norm=lambda a: torch.from_numpy(((a-mu)/sd).astype("float32"))
    return norm(x[tr]),norm(x[te]),torch.from_numpy(y[tr]),torch.from_numpy(y[te]),int(y.max()+1)

class MLP(nn.Module):
    def __init__(self,d,h,c,kind):
        super().__init__(); act={"linear":nn.Identity(),"relu":nn.ReLU(),"gelu":nn.GELU()}[kind]; self.net=nn.Sequential(nn.Linear(d,h),act,nn.Linear(h,c))
    def forward(self,x): return self.net(x.flatten(1))

def run(task,kind,a):
    xtr,xte,ytr,yte,c=make_data(task,a.n,a.seed); m=MLP(xtr.shape[1],a.hidden,c,kind).to(a.device); opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=1e-4)
    tr=DataLoader(TensorDataset(xtr,ytr),a.batch,shuffle=True); te=DataLoader(TensorDataset(xte,yte),1024); hist=[]; t0=time.perf_counter()
    for ep in range(1,a.epochs+1):
        m.train(); good=tot=loss_sum=0.
        for xb,yb in tr:
            xb,yb=xb.to(a.device),yb.to(a.device); z=m(xb); loss=F.cross_entropy(z,yb); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); loss_sum+=loss.item()*yb.numel(); good+=(z.argmax(1)==yb).sum().item(); tot+=yb.numel()
        m.eval(); goodt=nt=0
        with torch.no_grad():
            for xb,yb in te:
                z=m(xb.to(a.device)); goodt+=(z.argmax(1)==yb.to(a.device)).sum().item(); nt+=yb.numel()
        hist.append({"epoch":ep,"train_acc":good/tot,"test_acc":goodt/nt,"train_loss":loss_sum/tot})
    sec=time.perf_counter()-t0; return {"task":task,"model":kind,"parameters":sum(p.numel() for p in m.parameters()),"seconds":sec,"best_test_acc":max(h["test_acc"] for h in hist),"final_test_acc":hist[-1]["test_acc"],"history":hist}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--tasks",default="xor,circles,moons,checkerboard,parity8,noisy_moons100,noisy_spiral100"); p.add_argument("--epochs",type=int,default=100); p.add_argument("--n",type=int,default=6000); p.add_argument("--batch",type=int,default=128); p.add_argument("--hidden",type=int,default=64); p.add_argument("--lr",type=float,default=3e-3); p.add_argument("--seed",type=int,default=0); p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu"); p.add_argument("--result",default="bio_results/mlp_simple_complex_seed0.json"); a=p.parse_args(); os.makedirs(os.path.dirname(a.result) or ".",exist_ok=True); rows=[]
    for task in a.tasks.split(","):
        for kind in ("linear","relu","gelu"):
            torch.manual_seed(a.seed); r=run(task,kind,a); rows.append(r); print(f"[{task}] {kind:6s} best={r['best_test_acc']:.4f} final={r['final_test_acc']:.4f} time={r['seconds']:.1f}s",flush=True)
    json.dump(rows,open(a.result,"w",encoding="utf-8"),indent=2,ensure_ascii=False)
if __name__=="__main__": main()
