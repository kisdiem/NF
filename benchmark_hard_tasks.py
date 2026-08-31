"""Harder-than-sanity-check benchmark for MLP vs Local Electrical NF."""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset
from local_electrical_nf import LocalElectricalField
from local_electrical_nf_v3 import LocalElectricalFieldV3

class MLP(nn.Module):
    def __init__(self, dim, classes, act):
        super().__init__(); self.a=nn.Linear(dim,64); self.b=nn.Linear(64,classes); self.act=act
    def forward(self,x): return self.b(self.act(self.a(x)))

class LocalNF(nn.Module):
    def __init__(self, dim, classes, no_inhibition=False):
        super().__init__(); self.a=nn.Linear(dim,64)
        if no_inhibition: self.field=LocalElectricalField(8,8,4,threshold_init=.5,strength_init=.5,decay_init=.8,tau=.2,no_threshold=False,persistence=True,inhibition=False)
        else: self.field=LocalElectricalFieldV3(8,8,4,mode='raw_bounded',fuse_local_convs=True,collect_diagnostics=False)
        self.b=nn.Linear(64,classes)
    def forward(self,x): return self.b(self.field(self.a(x).view(-1,1,8,8)).flatten(1))

def make_data(name,n,seed):
    rng=np.random.RandomState(seed)
    if name=='spiral3':
        per=n//3; xs=[]; ys=[]
        for c in range(3):
            r=np.linspace(.05,1,per); th=np.linspace(c*2*np.pi/3,(c+1)*2*np.pi/3,per)+rng.randn(per)*.18; xs.append(np.c_[r*np.cos(th),r*np.sin(th)]); ys.extend([c]*per)
        x=np.concatenate(xs).astype('float32'); y=np.array(ys,'int64')
    elif name=='checkerboard':
        x=rng.uniform(-1,1,(n,2)).astype('float32'); bins=np.floor((x+1)*4).astype('int64'); y=((bins[:,0]+bins[:,1])%2).astype('int64'); x+=rng.randn(n,2).astype('float32')*.025
    elif name=='parity8':
        x=rng.choice([-1.,1.],size=(n,8)).astype('float32'); y=((x>0).sum(1)%2).astype('int64')
    elif name=='noisy_moons100':
        sig,y=make_moons(n_samples=n,noise=.20,random_state=seed); x=np.concatenate([sig.astype('float32'),rng.randn(n,98).astype('float32')],1); y=y.astype('int64')
    elif name=='noisy_spiral100':
        base_rng=np.random.RandomState(seed); per=n//3; xs=[]; ys=[]
        for c in range(3):
            r=np.linspace(.05,1,per); th=np.linspace(c*2*np.pi/3,(c+1)*2*np.pi/3,per)+base_rng.randn(per)*.18; xs.append(np.c_[r*np.cos(th),r*np.sin(th)]); ys.extend([c]*per)
        sig=np.concatenate(xs).astype('float32'); x=np.concatenate([sig,base_rng.randn(len(sig),98).astype('float32')],1); y=np.array(ys,'int64')
    else: raise ValueError(name)
    order=rng.permutation(len(y)); split=int(.8*len(y)); tr,te=order[:split],order[split:]
    mu=x[tr].mean(0,keepdims=True); sd=x[tr].std(0,keepdims=True)+1e-6; xtr=(x[tr]-mu)/sd; xte=(x[te]-mu)/sd
    return TensorDataset(torch.tensor(xtr),torch.tensor(y[tr])),TensorDataset(torch.tensor(xte),torch.tensor(y[te])),x.shape[1],int(y.max()+1)

def evaluate(m,loader):
    m.eval(); c=t=0
    with torch.inference_mode():
        for x,y in loader: c+=(m(x).argmax(1)==y).sum().item(); t+=y.numel()
    return c/t

def run(kind,tr,te,dim,classes,a):
    torch.manual_seed(a.seed); m=MLP(dim,classes,F.relu if kind=='relu' else F.gelu) if kind in ('relu','gelu') else LocalNF(dim,classes,kind=='no_inhibition'); train=DataLoader(tr,batch_size=a.batch,shuffle=True); test=DataLoader(te,batch_size=1024); opt=torch.optim.Adam(m.parameters(),lr=a.lr,weight_decay=1e-4); hist=[]; st=time.perf_counter()
    for ep in range(a.epochs):
        m.train(); c=t=0
        for x,y in train:
            z=m(x); loss=F.cross_entropy(z,y); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step(); c+=(z.argmax(1)==y).sum().item(); t+=y.numel()
        hist.append({'epoch':ep+1,'train_acc':c/t,'test_acc':evaluate(m,test)})
    return {'model':kind,'best_test_acc':max(r['test_acc'] for r in hist),'final_test_acc':hist[-1]['test_acc'],'history':hist,'parameters':sum(p.numel() for p in m.parameters()),'seconds':time.perf_counter()-st}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--tasks',default='spiral3,checkerboard,parity8,noisy_moons100,noisy_spiral100'); p.add_argument('--n',type=int,default=6000); p.add_argument('--epochs',type=int,default=100); p.add_argument('--batch',type=int,default=128); p.add_argument('--lr',type=float,default=3e-3); p.add_argument('--seed',type=int,default=0); a=p.parse_args(); torch.set_num_threads(1); out='hard_task_results'; os.makedirs(out,exist_ok=True); allr={}
    for task in a.tasks.split(','):
        tr,te,dim,classes=make_data(task,a.n,a.seed); allr[task]=[run(k,tr,te,dim,classes,a) for k in ('relu','gelu','raw_bounded','no_inhibition')]; print('\n'+task); [print(f"{r['model']:14s} best={r['best_test_acc']:.4f} final={r['final_test_acc']:.4f} params={r['parameters']} time={r['seconds']:.2f}s") for r in allr[task]]
    with open(os.path.join(out,'results_seed0.json'),'w',encoding='utf-8') as f: json.dump(allr,f,indent=2)
if __name__=='__main__': main()
