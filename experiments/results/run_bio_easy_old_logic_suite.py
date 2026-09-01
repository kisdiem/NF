"""Compare bio-old (full dynamics) and bio-eazy (one-step static) on a logic suite."""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import make_moons, make_circles
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from models.bio_neuron import BioMLP


def raw(name, n, seed):
    rng = np.random.RandomState(seed)
    if name == "xor":
        x=rng.choice([-1.,1.],(n,2)).astype("float32"); x+=rng.randn(n,2).astype("float32")*.08; y=((x[:,0]*x[:,1])<0).astype("int64")
    elif name == "circles": x,y=make_circles(n,noise=.12,factor=.5,random_state=seed); x=x.astype("float32"); y=y.astype("int64")
    elif name == "moons": x,y=make_moons(n,noise=.18,random_state=seed); x=x.astype("float32"); y=y.astype("int64")
    elif name == "checkerboard":
        x=rng.uniform(-1,1,(n,2)).astype("float32"); b=np.floor((x+1)*4).astype("int64"); y=((b[:,0]+b[:,1])%2).astype("int64"); x+=rng.randn(n,2).astype("float32")*.025
    elif name == "parity8": x=rng.choice([-1.,1.],(n,8)).astype("float32"); y=((x>0).sum(1)%2).astype("int64")
    elif name == "noisy_moons100":
        sig,y=make_moons(n,noise=.20,random_state=seed); x=np.c_[sig.astype("float32"),rng.randn(n,98).astype("float32")]
    elif name == "noisy_spiral100":
        base=np.random.RandomState(seed); per=n//3; xs=[]; ys=[]
        for c in range(3):
            r=np.linspace(.05,1,per); th=np.linspace(c*2*np.pi/3,(c+1)*2*np.pi/3,per)+base.randn(per)*.18; xs.append(np.c_[r*np.cos(th),r*np.sin(th)]); ys.extend([c]*per)
        sig=np.concatenate(xs).astype("float32"); x=np.c_[sig,base.randn(len(sig),98).astype("float32")]; y=np.array(ys,"int64")
    elif name in ("hierarchical_xor16","variable_parity16"):
        x=rng.choice([-1.,1.],(n,16)).astype("float32")
        if name.startswith("variable"):
            lengths=rng.choice([4,8,12,16],n); mask=np.arange(16)[None,:]<lengths[:,None]; x[~mask]=0.; y=((x>0).sum(1)%2).astype("int64")
        else: y=((x>0).sum(1)%2).astype("int64")
    elif name == "multiplexer":
        addr=rng.randint(0,8,n); data=rng.choice([-1.,1.],(n,8)).astype("float32"); x=np.c_[((addr[:,None]>>np.arange(3))&1)*2-1,data]; y=(data[np.arange(n),addr]>0).astype("int64")
    elif name == "cnf_dnf":
        x=rng.choice([-1.,1.],(n,8)).astype("float32"); b=x>0; y=(((b[:,0]&b[:,1]&~b[:,2]) | (b[:,3]&~b[:,4]) | (~b[:,5]&b[:,6]&b[:,7]))).astype("int64")
    elif name == "carry8":
        a=rng.randint(0,256,n); b=rng.randint(0,256,n); bits=lambda z: ((z[:,None]>>np.arange(8))&1)*2-1; x=np.c_[bits(a),bits(b)].astype("float32"); y=((a+b)>=256).astype("int64")
    elif name == "equality16":
        y=rng.randint(0,2,n); a=rng.choice([-1.,1.],(n,16)).astype("float32"); b=a.copy(); bad=(y==0); flips=rng.randint(0,16,bad.sum()); b[bad,np.arange(16)[flips]]= -b[bad,np.arange(16)[flips]]; x=np.c_[a,b];
    elif name == "majority32":
        x=rng.choice([-1.,1.],(n,32)).astype("float32"); y=((x>0).sum(1)>16).astype("int64")
    elif name == "delayed_xor20":
        x=rng.choice([-1.,1.],(n,20)).astype("float32"); y=(x[:,3]*x[:,15]<0).astype("int64")
    elif name == "fsm_sequence":
        symbols=rng.randint(0,3,(n,12)); state=np.zeros(n,dtype="int64"); table=np.array([[0,1,2],[1,2,0],[2,0,1]])
        for t in range(12): state=table[state,symbols[:,t]]
        x=np.zeros((n,36),dtype="float32"); x[np.arange(n)[:,None],symbols+3*np.arange(12)[None,:]]=1.; y=state
    else: raise ValueError(name)
    return x.astype("float32"), np.asarray(y,"int64")


def data(name,n,seed):
    xt,yt=raw(name,n,seed); xv,yv=raw(name,n,seed+10000); mu=xt.mean(0,keepdims=True); sd=xt.std(0,keepdims=True)+1e-6
    return torch.from_numpy((xt-mu)/sd),torch.from_numpy((xv-mu)/sd),torch.from_numpy(yt),torch.from_numpy(yv),int(max(yt.max(),yv.max())+1)


def run(name, kind, seed, args):
    xtr,xte,ytr,yte,classes=data(name,args.n,seed); d=xtr.shape[1]
    if kind=="bio-old": cfg=dict(branches=4,steps=3,temporal=True,membrane_decay=.8)
    else: cfg=dict(branches=4,steps=1,temporal=False,membrane_decay=0.)
    model=BioMLP(d,64,classes,dendrite="soft_threshold",inhibition=True,adaptive_threshold=True,output_mode="mean",**cfg).to(args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4); tr=DataLoader(TensorDataset(xtr,ytr),args.batch,shuffle=True); te=DataLoader(TensorDataset(xte,yte),1024); hist=[]; start=time.perf_counter()
    for ep in range(1,args.epochs+1):
        model.train(); good=total=loss_sum=0.
        for xb,yb in tr:
            xb,yb=xb.to(args.device),yb.to(args.device); z=model(xb); loss=F.cross_entropy(z,yb); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step(); loss_sum+=loss.item()*yb.numel(); good+=(z.argmax(1)==yb).sum().item(); total+=yb.numel()
        model.eval(); goodt=nt=0
        with torch.no_grad():
            for xb,yb in te:
                z=model(xb.to(args.device)); goodt+=(z.argmax(1)==yb.to(args.device)).sum().item(); nt+=yb.numel()
        hist.append({"epoch":ep,"train_acc":good/total,"test_acc":goodt/nt,"train_loss":loss_sum/total})
        if ep==1 or ep%10==0 or ep==args.epochs: print(f"[{name} {kind}] ep={ep:03d} train={good/total:.4f} test={goodt/nt:.4f}",flush=True)
    sec=time.perf_counter()-start; return {"task":name,"model":kind,"seed":seed,"parameters":sum(p.numel() for p in model.parameters()),"seconds":sec,"best_test_acc":max(h['test_acc'] for h in hist),"final_test_acc":hist[-1]['test_acc'],"history":hist}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--tasks",default="xor,circles,moons,checkerboard,parity8,noisy_moons100,noisy_spiral100,hierarchical_xor16,multiplexer,cnf_dnf,carry8,variable_parity16,equality16,majority32,delayed_xor20,fsm_sequence"); p.add_argument("--epochs",type=int,default=100); p.add_argument("--n",type=int,default=6000); p.add_argument("--batch",type=int,default=128); p.add_argument("--lr",type=float,default=3e-3); p.add_argument("--seed",type=int,default=0); p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu"); p.add_argument("--result",required=True); a=p.parse_args(); rows=[]
    for task in a.tasks.split(","):
        for kind in ("bio-old","bio-eazy"):
            torch.manual_seed(a.seed); rows.append(run(task,kind,a.seed,a))
            with open(a.result,"w",encoding="utf-8") as f: json.dump({"config":vars(a),"results":rows},f,indent=2,ensure_ascii=False)
    print("SUMMARY")
    for task in a.tasks.split(","):
        rr=[r for r in rows if r['task']==task]; print(task,[(r['model'],round(r['best_test_acc']*100,2),round(r['seconds'],1)) for r in rr])

if __name__=="__main__": main()
