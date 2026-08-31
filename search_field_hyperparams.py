"""Validate whether Local NF parameters work better as hyperparameters.

For each candidate, field parameters are fixed and only the two linear maps
are trained.  Candidates are selected by a validation split, then the chosen
configuration is retrained on all training data and evaluated once on test.
"""
import argparse, json, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from benchmark_hard_tasks import make_data
from local_electrical_nf_v3 import LocalElectricalFieldV3


class FixedFieldModel(nn.Module):
    def __init__(self, dim, classes, cfg):
        super().__init__(); self.input=nn.Linear(dim,64); self.field=LocalElectricalFieldV3(8,8,cfg['steps'],mode='raw_bounded',threshold_init=cfg['threshold'],strength_init=cfg['strength'],decay_init=cfg['decay'],rho_init=cfg['rho'],beta_init=cfg['beta'],collect_diagnostics=False); self.output=nn.Linear(64,classes)
        for p in self.field.parameters(): p.requires_grad=False
    def forward(self,x): return self.output(self.field(self.input(x).view(-1,1,8,8)).flatten(1))


def acc(model, loader):
    model.eval(); c=t=0
    with torch.inference_mode():
        for x,y in loader: c+=(model(x).argmax(1)==y).sum().item(); t+=y.numel()
    return c/t


def fit(model, loader, val_loader, epochs, lr):
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=1e-4); best=-1.; best_state=None; hist=[]
    for ep in range(epochs):
        model.train(); c=t=0
        for x,y in loader:
            z=model(x); loss=F.cross_entropy(z,y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); c+=(z.argmax(1)==y).sum().item(); t+=y.numel()
        v=acc(model,val_loader); hist.append({'epoch':ep+1,'train_acc':c/t,'val_acc':v})
        if v>best: best=v; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state); return best,hist


def main():
    p=argparse.ArgumentParser(); p.add_argument('--tasks',default='checkerboard,noisy_moons100,noisy_spiral100'); p.add_argument('--n',type=int,default=6000); p.add_argument('--search-epochs',type=int,default=15); p.add_argument('--final-epochs',type=int,default=20); p.add_argument('--batch',type=int,default=128); p.add_argument('--lr',type=float,default=3e-3); p.add_argument('--seed',type=int,default=0); p.add_argument('--result-tag',default='seed0'); a=p.parse_args(); torch.set_num_threads(1); torch.manual_seed(a.seed)
    candidates=[
        {'name':'default','threshold':.5,'strength':.5,'decay':.8,'steps':4,'beta':1.,'rho':.15},
        {'name':'low_threshold','threshold':.25,'strength':.5,'decay':.8,'steps':4,'beta':1.,'rho':.15},
        {'name':'high_threshold','threshold':.9,'strength':.5,'decay':.8,'steps':4,'beta':1.,'rho':.15},
        {'name':'low_strength','threshold':.5,'strength':.25,'decay':.8,'steps':4,'beta':1.,'rho':.15},
        {'name':'high_strength','threshold':.5,'strength':1.,'decay':.8,'steps':4,'beta':1.,'rho':.15},
        {'name':'low_decay','threshold':.5,'strength':.5,'decay':.6,'steps':4,'beta':1.,'rho':.15},
        {'name':'high_decay','threshold':.5,'strength':.5,'decay':.95,'steps':4,'beta':1.,'rho':.15},
        {'name':'two_steps','threshold':.5,'strength':.5,'decay':.8,'steps':2,'beta':1.,'rho':.15},
        {'name':'six_steps','threshold':.5,'strength':.5,'decay':.8,'steps':6,'beta':1.,'rho':.15},
        {'name':'weak_inhibition','threshold':.5,'strength':.5,'decay':.8,'steps':4,'beta':.25,'rho':.15},
        {'name':'strong_inhibition','threshold':.5,'strength':.5,'decay':.8,'steps':4,'beta':2.,'rho':.15},
        {'name':'late_inhibition','threshold':.5,'strength':.5,'decay':.8,'steps':4,'beta':1.,'rho':.5},
    ]
    all_results={}; start=time.perf_counter()
    for task in a.tasks.split(','):
        tr,te,dim,classes=make_data(task,a.n,a.seed); gen=torch.Generator().manual_seed(a.seed); tr_fit,tr_val=random_split(tr,[int(.8*len(tr)),len(tr)-int(.8*len(tr))],generator=gen); fit_loader=DataLoader(tr_fit,a.batch,shuffle=True); val_loader=DataLoader(tr_val,1024); test_loader=DataLoader(te,1024)
        rows=[]
        for cfg in candidates:
            torch.manual_seed(a.seed); m=FixedFieldModel(dim,classes,cfg); best_val,hist=fit(m,fit_loader,val_loader,a.search_epochs,a.lr); rows.append({'config':cfg,'best_val_acc':best_val,'search_history':hist}); print(f"{task:18s} {cfg['name']:18s} val={best_val:.4f}")
        selected=max(rows,key=lambda r:r['best_val_acc'])
        # Retrain selected fixed field on the complete training split.
        torch.manual_seed(a.seed+17); final=FixedFieldModel(dim,classes,selected['config']); best_test_history=[]; opt=torch.optim.Adam([p for p in final.parameters() if p.requires_grad],lr=a.lr,weight_decay=1e-4); full_loader=DataLoader(tr,a.batch,shuffle=True)
        for ep in range(a.final_epochs):
            final.train()
            for x,y in full_loader:
                z=final(x); loss=F.cross_entropy(z,y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            best_test_history.append(acc(final,test_loader))
        all_results[task]={'candidates':rows,'selected_by_validation':selected['config'],'selected_validation_acc':selected['best_val_acc'],'selected_test_best':max(best_test_history),'selected_test_final':best_test_history[-1],'selected_test_history':best_test_history}
        print(f"{task:18s} SELECTED {selected['config']['name']:18s} val={selected['best_val_acc']:.4f} test_best={max(best_test_history):.4f}")
    result={'config':vars(a),'tasks':all_results,'seconds_total':time.perf_counter()-start}; out='local_electrical_nf_usage_results'; os.makedirs(out,exist_ok=True); path=os.path.join(out,'field_hyperparam_search_'+a.result_tag+'.json');
    with open(path,'w',encoding='utf-8') as f: json.dump(result,f,indent=2)
    print('saved',path)


if __name__=='__main__': main()
