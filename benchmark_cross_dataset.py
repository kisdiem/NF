"""Cross-dataset benchmark for Linear/ReLU/GELU and Local Electrical NF.

Synthetic tasks use the same 64 hidden units and the same optimizer.  Local NF
projects each input dimension to an 8x8 field, so the field is tested as a
nonlinear module rather than as an image-specific input layer.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits, make_circles, make_classification, make_moons
from torch.utils.data import DataLoader, TensorDataset
from local_electrical_nf import LocalElectricalField
from local_electrical_nf_v3 import LocalElectricalFieldV3


class PlainMLP(nn.Module):
    def __init__(self, dim, classes, activation=None):
        super().__init__(); self.fc1=nn.Linear(dim,64); self.fc2=nn.Linear(64,classes); self.activation=activation
    def forward(self,x):
        h=self.fc1(x); return self.fc2(self.activation(h) if self.activation else h)


class LocalSyntheticNF(nn.Module):
    def __init__(self, dim, classes, variant="raw_bounded"):
        super().__init__(); self.fc1=nn.Linear(dim,64)
        if variant == "no_inhibition":
            self.field=LocalElectricalField(8,8,4,threshold_init=.5,strength_init=.5,decay_init=.8,tau=.2,no_threshold=False,persistence=True,inhibition=False)
        else:
            self.field=LocalElectricalFieldV3(8,8,4,mode="raw_bounded",fuse_local_convs=True)
        self.fc2=nn.Linear(64,classes)
    def forward(self,x): return self.fc2(self.field(self.fc1(x).view(-1,1,8,8)).flatten(1))


def dataset(name, n, seed):
    rng=np.random.RandomState(seed)
    if name == "digits8x8":
        digits = load_digits(); x=digits.data.astype("float32")/16.0; y=digits.target.astype("int64")
        order=rng.permutation(len(y)); x=x[order[:min(n,len(y))]]; y=y[order[:min(n,len(y))]]
        n=len(y)
    elif name == "xor":
        x=rng.uniform(-1,1,(n,2)).astype("float32"); y=((x[:,0]*x[:,1])<0).astype("int64")
    elif name == "circles": x,y=make_circles(n_samples=n,noise=.12,factor=.5,random_state=seed); x=x.astype("float32"); y=y.astype("int64")
    elif name == "moons": x,y=make_moons(n_samples=n,noise=.18,random_state=seed); x=x.astype("float32"); y=y.astype("int64")
    elif name == "classification10": x,y=make_classification(n_samples=n,n_features=10,n_informative=6,n_redundant=2,n_classes=2,class_sep=1.0,random_state=seed); x=x.astype("float32"); y=y.astype("int64")
    else: raise ValueError(name)
    split=int(.8*n); order=rng.permutation(n); tr,te=order[:split],order[split:]
    # Fit preprocessing on the training split only.
    mu=x[tr].mean(0,keepdims=True); sigma=x[tr].std(0,keepdims=True)+1e-6
    x_train=(x[tr]-mu)/sigma; x_test=(x[te]-mu)/sigma
    return TensorDataset(torch.tensor(x_train),torch.tensor(y[tr])), TensorDataset(torch.tensor(x_test),torch.tensor(y[te])), int(y.max()+1), x.shape[1]


def evaluate(model, loader):
    model.eval(); c=n=0
    with torch.inference_mode():
        for x,y in loader: c+=(model(x).argmax(1)==y).sum().item(); n+=y.numel()
    return c/n


def run(model_name, tr, te, dim, classes, args):
    torch.manual_seed(args.seed)
    if model_name=="linear": model=PlainMLP(dim,classes)
    elif model_name=="relu": model=PlainMLP(dim,classes,F.relu)
    elif model_name=="gelu": model=PlainMLP(dim,classes,F.gelu)
    else: model=LocalSyntheticNF(dim,classes,model_name)
    train=DataLoader(tr,batch_size=args.batch,shuffle=True); test=DataLoader(te,batch_size=1024)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-4); hist=[]; start=time.perf_counter()
    for ep in range(args.epochs):
        model.train(); c=n=0
        for x,y in train:
            z=model(x); loss=F.cross_entropy(z,y); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); c+=(z.argmax(1)==y).sum().item(); n+=y.numel()
        hist.append({"epoch":ep+1,"train_acc":c/n,"test_acc":evaluate(model,test)})
    return {"model":model_name,"best_test_acc":max(r["test_acc"] for r in hist),"final_test_acc":hist[-1]["test_acc"],"history":hist,"parameters":sum(p.numel() for p in model.parameters()),"seconds":time.perf_counter()-start}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--datasets",default="xor,circles,moons,classification10"); p.add_argument("--n",type=int,default=4000); p.add_argument("--epochs",type=int,default=50); p.add_argument("--batch",type=int,default=128); p.add_argument("--lr",type=float,default=3e-3); p.add_argument("--seed",type=int,default=0); p.add_argument("--result-tag",default="seed0"); args=p.parse_args(); torch.set_num_threads(1)
    out="cross_dataset_results"; os.makedirs(out,exist_ok=True); all_results={}
    for name in args.datasets.split(','):
        tr,te,classes,dim=dataset(name,args.n,args.seed); models=["linear","relu","gelu","raw_bounded","no_inhibition"]; all_results[name]=[run(m,tr,te,dim,classes,args) for m in models]
        print("\n",name)
        for r in all_results[name]: print(f"{r['model']:14s} best={r['best_test_acc']:.4f} final={r['final_test_acc']:.4f} params={r['parameters']} time={r['seconds']:.2f}s")
    with open(os.path.join(out,"results_"+args.result_tag+".json"),"w",encoding="utf-8") as f: json.dump(all_results,f,indent=2)

if __name__=="__main__": main()
