"""Short server experiment: replace only the final habitat MLP GELU by Local NF."""
import argparse, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from local_electrical_nf_v3 import LocalElectricalFieldV3


class HabitatMLP(nn.Module):
    def __init__(self, d):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d,32),nn.LayerNorm(32),nn.GELU(),nn.Dropout(.3),nn.Linear(32,16),nn.LayerNorm(16),nn.GELU(),nn.Dropout(.3),nn.Linear(16,1))
    def forward(self,x): return self.net(x).squeeze(-1)


class HabitatNF(nn.Module):
    def __init__(self,d,steps=4):
        super().__init__(); self.pre=nn.Sequential(nn.Linear(d,32),nn.LayerNorm(32),nn.GELU(),nn.Dropout(.3),nn.Linear(32,16),nn.LayerNorm(16)); self.field=LocalElectricalFieldV3(4,4,steps,mode='raw_bounded',threshold_init=.5,strength_init=.5,decay_init=.8,collect_diagnostics=False); self.out=nn.Sequential(nn.Dropout(.3),nn.Linear(16,1))
    def forward(self,x): return self.out(self.field(self.pre(x).view(-1,1,4,4)).flatten(1)).squeeze(-1)


def metrics(model,loader,device):
    model.eval(); c=t=0; losses=[]
    with torch.inference_mode():
        for x,y in loader:
            z=model(x.to(device)).reshape(-1); y=y.to(device).reshape(-1); losses.append(F.binary_cross_entropy_with_logits(z,y).item()); c+=((z>0)==(y>.5)).sum().item(); t+=y.numel()
    return {'acc':c/t,'loss':float(np.mean(losses))}


def run(model,train_loader,val_loader,epochs,device,pos_weight):
    model.to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3); hist=[]
    for ep in range(epochs):
        model.train()
        for x,y in train_loader:
            x,y=x.to(device),y.to(device).reshape(-1); z=model(x).reshape(-1); loss=F.binary_cross_entropy_with_logits(z,y,pos_weight=pos_weight); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        hist.append({'epoch':ep+1,'train':metrics(model,train_loader,device),'val':metrics(model,val_loader,device)})
        print(ep+1,hist[-1],flush=True)
    return hist


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--epochs',type=int,default=5); p.add_argument('--batch',type=int,default=128); p.add_argument('--seed',type=int,default=3866); p.add_argument('--device',default='cuda'); p.add_argument('--out',required=True); a=p.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); device=torch.device(a.device if torch.cuda.is_available() else 'cpu')
    records=json.load(open(a.data)); x=np.asarray([r['habitat_features'] for r in records],dtype='float32'); y=np.asarray([r['label'] for r in records],dtype='float32').reshape(-1); rng=np.random.RandomState(a.seed); train_idx=[]; val_idx=[]
    for label in (0.,1.):
        idx=np.where(y==label)[0]; rng.shuffle(idx); cut=max(1,int(len(idx)*.8)); train_idx.extend(idx[:cut]); val_idx.extend(idx[cut:])
    mu=np.nanmean(x[train_idx],axis=0); sd=np.nanstd(x[train_idx],axis=0)+1e-6; x=np.nan_to_num((x-mu)/sd,nan=0.,posinf=0.,neginf=0.); tr=TensorDataset(torch.from_numpy(x[train_idx]),torch.from_numpy(y[train_idx])); va=TensorDataset(torch.from_numpy(x[val_idx]),torch.from_numpy(y[val_idx])); train_loader=DataLoader(tr,a.batch,shuffle=True); val_loader=DataLoader(va,1024); pos=float((y[train_idx]==0).sum()/max(1,(y[train_idx]==1).sum())); pos_weight=torch.tensor(pos,device=device)
    results={}; start=time.perf_counter()
    for name,ctor in [('mlp',lambda:HabitatMLP(x.shape[1])),('nf',lambda:HabitatNF(x.shape[1],4))]:
        torch.manual_seed(a.seed); model=ctor(); hist=run(model,train_loader,val_loader,a.epochs,device,pos_weight); results[name]={'parameters':sum(p.numel() for p in model.parameters()),'history':hist,'best_val_acc':max(r['val']['acc'] for r in hist)}
    results['config']=vars(a); results['seconds']=time.perf_counter()-start
    with open(a.out,'w') as f: json.dump(results,f,indent=2); print(json.dumps(results,indent=2))


if __name__=='__main__': main()
