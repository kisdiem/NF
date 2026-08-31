"""Replace every GELU in HabitatOnlyMLP with a local NF field."""
import argparse, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from local_electrical_nf_v3 import LocalElectricalFieldV3


class OriginalMLP(nn.Module):
    def __init__(self,d):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d,32),nn.LayerNorm(32),nn.GELU(),nn.Dropout(.3),nn.Linear(32,16),nn.LayerNorm(16),nn.GELU(),nn.Dropout(.3),nn.Linear(16,1))
    def forward(self,x): return self.net(x).reshape(-1)


class FullNF(nn.Module):
    def __init__(self,d,cfg):
        super().__init__(); self.l1=nn.Linear(d,32); self.n1=nn.LayerNorm(32); self.f1=LocalElectricalFieldV3(4,8,cfg['steps1'],mode='raw_bounded',threshold_init=cfg['threshold'],strength_init=cfg['strength'],decay_init=cfg['decay'],beta_init=cfg['beta'],rho_init=cfg['rho'],collect_diagnostics=False); self.drop1=nn.Dropout(.3); self.l2=nn.Linear(32,16); self.n2=nn.LayerNorm(16); self.f2=LocalElectricalFieldV3(4,4,cfg['steps2'],mode='raw_bounded',threshold_init=cfg['threshold'],strength_init=cfg['strength'],decay_init=cfg['decay'],beta_init=cfg['beta'],rho_init=cfg['rho'],collect_diagnostics=False); self.drop2=nn.Dropout(.3); self.out=nn.Linear(16,1)
    def forward(self,x):
        x=self.drop1(self.f1(self.n1(self.l1(x)).view(-1,1,4,8)).flatten(1)); x=self.drop2(self.f2(self.n2(self.l2(x)).view(-1,1,4,4)).flatten(1)); return self.out(x).reshape(-1)


def metrics(model,loader,device):
    model.eval(); c=t=0; losses=[]; ps=[]; ys=[]
    with torch.inference_mode():
        for x,y in loader:
            z=model(x.to(device)).reshape(-1); y=y.to(device).reshape(-1); losses.append(F.binary_cross_entropy_with_logits(z,y).item()); c+=((z>0)==(y>.5)).sum().item(); t+=y.numel(); ps.extend(torch.sigmoid(z).cpu().tolist()); ys.extend(y.cpu().tolist())
    return {'acc':c/t,'loss':float(np.mean(losses)),'auc':float(roc_auc_score(ys,ps))}


def run(model,train_loader,val_loader,epochs,device,pos_weight):
    model.to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3); hist=[]
    for ep in range(epochs):
        model.train()
        for x,y in train_loader:
            x,y=x.to(device),y.to(device).reshape(-1); z=model(x); loss=F.binary_cross_entropy_with_logits(z,y,pos_weight=pos_weight); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        row={'epoch':ep+1,'train':metrics(model,train_loader,device),'val':metrics(model,val_loader,device)}; hist.append(row); print(ep+1,model.__class__.__name__,hist[-1],flush=True)
    return hist


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--epochs',type=int,default=30); p.add_argument('--batch',type=int,default=128); p.add_argument('--seed',type=int,default=3866); p.add_argument('--device',default='cuda'); p.add_argument('--out',required=True); a=p.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); device=torch.device(a.device if torch.cuda.is_available() else 'cpu')
    records=json.load(open(a.data)); x=np.asarray([r['habitat_features'] for r in records],dtype='float32'); y=np.asarray([r['label'] for r in records],dtype='float32').reshape(-1); rng=np.random.RandomState(a.seed); train_idx=[]; val_idx=[]
    for label in (0.,1.):
        idx=np.where(y==label)[0]; rng.shuffle(idx); cut=max(1,int(len(idx)*.8)); train_idx.extend(idx[:cut]); val_idx.extend(idx[cut:])
    mu=np.nanmean(x[train_idx],axis=0); sd=np.nanstd(x[train_idx],axis=0)+1e-6; x=np.nan_to_num((x-mu)/sd,nan=0.,posinf=0.,neginf=0.); tr=TensorDataset(torch.from_numpy(x[train_idx]),torch.from_numpy(y[train_idx])); va=TensorDataset(torch.from_numpy(x[val_idx]),torch.from_numpy(y[val_idx])); train_loader=DataLoader(tr,a.batch,shuffle=True); val_loader=DataLoader(va,1024); pos=float((y[train_idx]==0).sum()/max(1,(y[train_idx]==1).sum())); pos_weight=torch.tensor(pos,device=device)
    configs={'conservative':{'threshold':.5,'strength':.25,'decay':.6,'steps1':2,'steps2':2,'beta':1.,'rho':.15},'default':{'threshold':.5,'strength':.5,'decay':.8,'steps1':4,'steps2':4,'beta':1.,'rho':.15},'short_fast':{'threshold':.35,'strength':.25,'decay':.8,'steps1':1,'steps2':1,'beta':.5,'rho':.15}}
    results={}; start=time.perf_counter()
    for name,cfg in [('mlp',None)]+list(configs.items()):
        torch.manual_seed(a.seed); model=OriginalMLP(x.shape[1]) if name=='mlp' else FullNF(x.shape[1],cfg); hist=run(model,train_loader,val_loader,a.epochs,device,pos_weight); results[name]={'config':cfg,'parameters':sum(p.numel() for p in model.parameters()),'best_val_auc':max(r['val']['auc'] for r in hist),'best_val_acc':max(r['val']['acc'] for r in hist),'history':hist}
    results['config_run']=vars(a); results['seconds']=time.perf_counter()-start
    with open(a.out,'w') as f: json.dump(results,f,indent=2); print(json.dumps(results,indent=2))


if __name__=='__main__': main()
