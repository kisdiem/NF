"""Core BioNeuron baselines, ablations and branch-count experiments."""
import argparse, json, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.bio_neuron import BioMLP


def synthetic(name, n=1024, seed=0):
    g = torch.Generator().manual_seed(seed)
    if name == "xor":
        base = torch.tensor([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
        x = base.repeat((n + 3)//4, 1)[:n] + 0.08 * torch.randn(n,2,generator=g)
        y = ((x[:,0] * x[:,1]) < 0).long()
    elif name == "circles":
        angles = 2 * torch.pi * torch.rand(n, generator=g)
        y = torch.arange(n) % 2
        radius = torch.where(y == 0, torch.tensor(0.65), torch.tensor(1.55)) + 0.08 * torch.randn(n,generator=g)
        x = torch.stack((radius*angles.cos(), radius*angles.sin()), 1)
    elif name == "moons":
        half=n//2; t=torch.pi*torch.rand(half,generator=g)
        x0=torch.stack((t.cos(),t.sin()),1); x1=torch.stack((1-t.cos(),0.35-t.sin()),1)
        x=torch.cat((x0,x1),0)+0.10*torch.randn(n,2,generator=g); y=torch.cat((torch.zeros(half),torch.ones(n-half))).long()
    else:
        raise ValueError(name)
    p=torch.randperm(n,generator=g); return x[p],y[p]


class Baseline(nn.Module):
    def __init__(self,d_in,hidden,d_out,kind):
        super().__init__(); self.fc1=nn.Linear(d_in,hidden); self.fc2=nn.Linear(hidden,d_out); self.kind=kind
    def forward(self,x):
        h=self.fc1(x.flatten(1))
        if self.kind=="relu": h=F.relu(h)
        elif self.kind=="gelu": h=F.gelu(h)
        return self.fc2(h)


def load_task(name,args):
    if name != "mnist":
        xtr,ytr=synthetic(name,args.samples,args.seed); xte,yte=synthetic(name,args.test_samples,args.seed+1)
        return xtr,ytr,xte,yte,2,2,args.synthetic_hidden
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    tf=transforms.ToTensor(); tr=datasets.MNIST(args.data_root,True,download=True,transform=tf); te=datasets.MNIST(args.data_root,False,download=True,transform=tf)
    if args.subset: tr=Subset(tr,range(args.subset))
    tl=DataLoader(tr,batch_size=2048,shuffle=False); vl=DataLoader(te,batch_size=2048,shuffle=False)
    xtr=torch.cat([x.flatten(1) for x,_ in tl]); ytr=torch.cat([y for _,y in tl]); xte=torch.cat([x.flatten(1) for x,_ in vl]); yte=torch.cat([y for _,y in vl])
    return xtr,ytr,xte,yte,784,10,args.hidden


def make_bio(d_in,hidden,d_out,args,branches=None,ablation=None):
    cfg=dict(branches=branches or args.branches,steps=args.steps,dendrite=args.dendrite,temporal=True,inhibition=True,adaptive_threshold=True,hard_spike=args.hard_spike,output_mode=args.bio_output,weight_rank=args.bio_rank)
    if ablation=="no_temporal": cfg["temporal"]=False
    elif ablation=="no_inhibition": cfg["inhibition"]=False
    elif ablation=="fixed_threshold": cfg["adaptive_threshold"]=False
    elif ablation=="one_step": cfg["steps"]=1
    elif ablation=="one_branch": cfg["branches"]=1
    return BioMLP(d_in,hidden,d_out,**cfg)


def param_count(m): return sum(p.numel() for p in m.parameters())


def matched_hidden(target,d_in,d_out):
    return max(1, round((target-d_out)/(d_in+d_out+1)))


def train(model,xtr,ytr,xte,yte,args):
    dev=torch.device(args.device); model=model.to(dev); xtr,ytr,xte,yte=xtr.to(dev),ytr.to(dev),xte.to(dev),yte.to(dev)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-4); hist=[]; clock_start=time.monotonic()
    gen_device="cuda" if dev.type=="cuda" else "cpu"; g=torch.Generator(device=gen_device).manual_seed(args.seed)
    for ep in range(args.epochs):
        model.train(); order=torch.randperm(xtr.shape[0],generator=g,device=dev); loss_sum=0.0
        for batch_start in range(0,xtr.shape[0],args.batch):
            ids=order[batch_start:batch_start+args.batch]; z=model(xtr[ids]); loss=F.cross_entropy(z,ytr[ids]); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); loss_sum += loss.item()*ids.numel()
        model.eval()
        with torch.no_grad():
            pred=model(xte); acc=(pred.argmax(1)==yte).float().mean().item(); test_loss=F.cross_entropy(pred,yte).item()
        hist.append({"epoch":ep+1,"train_loss":loss_sum/xtr.shape[0],"test_loss":test_loss,"test_acc":acc})
    return {"parameters":param_count(model),"best_test_acc":max(r["test_acc"] for r in hist),"final_test_acc":hist[-1]["test_acc"],"seconds":time.monotonic()-clock_start,"history":hist}


def run_task(name,args):
    xtr,ytr,xte,yte,d_in,d_out,hidden=load_task(name,args)
    torch.manual_seed(args.seed)
    probe=make_bio(d_in,hidden,d_out,args)
    bio_params=param_count(probe); base_hidden=matched_hidden(bio_params,d_in,d_out) if args.parameter_match else hidden

    if args.branch_sweep:
        specs=[(f"bio_b{b}","bio",{"branches":b}) for b in [int(x) for x in args.branch_sweep.split(',')]]
    elif args.ablation:
        specs=[("bio_full","bio",{})]+[("bio_"+a,"bio",{"ablation":a}) for a in ("one_branch","no_temporal","no_inhibition","fixed_threshold","one_step")]
    else:
        specs=[("linear","baseline",{"kind":"linear"}),("relu","baseline",{"kind":"relu"}),("gelu","baseline",{"kind":"gelu"}),("bio","bio",{})]

    rows=[]
    for label,family,cfg in specs:
        # Seed is reset before model construction so initialization is controlled across variants.
        torch.manual_seed(args.seed)
        if family=="baseline": model=Baseline(d_in,base_hidden,d_out,cfg["kind"])
        else: model=make_bio(d_in,hidden,d_out,args,branches=cfg.get("branches"),ablation=cfg.get("ablation"))
        r=train(model,xtr,ytr,xte,yte,args); r.update({"task":name,"model":label,"parameter_match":args.parameter_match}); rows.append(r)
        print(f"{name:8s} {label:16s} best={r['best_test_acc']:.4f} params={r['parameters']} time={r['seconds']:.1f}s")
    return rows


def main():
    p=argparse.ArgumentParser(); p.add_argument("--task",choices=["xor","circles","moons","mnist","all"],default="all"); p.add_argument("--epochs",type=int,default=200); p.add_argument("--lr",type=float,default=1e-2); p.add_argument("--steps",type=int,default=3); p.add_argument("--branches",type=int,default=4); p.add_argument("--branch-sweep",default=""); p.add_argument("--parameter-match",action="store_true"); p.add_argument("--ablation",action="store_true"); p.add_argument("--samples",type=int,default=1024); p.add_argument("--test-samples",type=int,default=1024); p.add_argument("--synthetic-hidden",type=int,default=16); p.add_argument("--hidden",type=int,default=64); p.add_argument("--batch",type=int,default=256); p.add_argument("--subset",type=int,default=5000); p.add_argument("--seed",type=int,default=0); p.add_argument("--dendrite",choices=["soft_threshold","quadratic","tanh"],default="soft_threshold"); p.add_argument("--bio-rank",type=int,default=0); p.add_argument("--bio-output",choices=["mean","final","membrane"],default="mean"); p.add_argument("--hard-spike",action="store_true"); p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu"); p.add_argument("--data-root",default=os.path.join(ROOT,"data","mnist")); p.add_argument("--result",default="results/new_experiment.json"); a=p.parse_args()
    tasks=["xor","circles","moons","mnist"] if a.task=="all" else [a.task]; out={}; original_lr=a.lr
    for t in tasks:
        a.lr=min(original_lr,3e-3) if t=="mnist" else original_lr; out[t]=run_task(t,a)
    os.makedirs(os.path.dirname(a.result) or ".",exist_ok=True); json.dump({"config":vars(a),"results":out},open(a.result,"w",encoding="utf-8"),indent=2); print("written",a.result)
if __name__=="__main__": main()
