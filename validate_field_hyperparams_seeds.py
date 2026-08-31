"""Multi-seed validation of task-selected fixed Local NF dynamics."""
import argparse, json, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from benchmark_hard_tasks import make_data
from local_electrical_nf_v3 import LocalElectricalFieldV3


SELECTED = {
    'checkerboard': {'threshold': .5, 'strength': .5, 'decay': .8, 'steps': 4, 'beta': .25, 'rho': .15},
    'noisy_moons100': {'threshold': .5, 'strength': .5, 'decay': .8, 'steps': 2, 'beta': 1., 'rho': .15},
    'noisy_spiral100': {'threshold': .5, 'strength': .5, 'decay': .8, 'steps': 6, 'beta': 1., 'rho': .15},
}
DEFAULT = {'threshold': .5, 'strength': .5, 'decay': .8, 'steps': 4, 'beta': 1., 'rho': .15}


class FixedModel(nn.Module):
    def __init__(self, dim, classes, cfg):
        super().__init__(); self.input=nn.Linear(dim,64); self.field=LocalElectricalFieldV3(8,8,cfg['steps'],mode='raw_bounded',threshold_init=cfg['threshold'],strength_init=cfg['strength'],decay_init=cfg['decay'],beta_init=cfg['beta'],rho_init=cfg['rho'],collect_diagnostics=False); self.output=nn.Linear(64,classes)
        for p in self.field.parameters(): p.requires_grad=False
    def forward(self,x): return self.output(self.field(self.input(x).view(-1,1,8,8)).flatten(1))


def acc(model, loader):
    model.eval(); c=t=0
    with torch.inference_mode():
        for x,y in loader: c+=(model(x).argmax(1)==y).sum().item(); t+=y.numel()
    return c/t


def run(dim,classes,tr,te,cfg,seed,epochs,lr,batch):
    torch.manual_seed(seed); model=FixedModel(dim,classes,cfg); train_loader=DataLoader(tr,batch_size=batch,shuffle=True); test_loader=DataLoader(te,batch_size=1024); opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=1e-4); best=0.; final=0.
    for _ in range(epochs):
        model.train()
        for x,y in train_loader:
            z=model(x); loss=F.cross_entropy(z,y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        final=acc(model,test_loader); best=max(best,final)
    return best,final


def main():
    p=argparse.ArgumentParser(); p.add_argument('--tasks',default='checkerboard,noisy_moons100,noisy_spiral100'); p.add_argument('--n',type=int,default=6000); p.add_argument('--epochs',type=int,default=20); p.add_argument('--batch',type=int,default=128); p.add_argument('--lr',type=float,default=3e-3); p.add_argument('--data-seed',type=int,default=0); p.add_argument('--model-seeds',default='0,1,2'); p.add_argument('--result-tag',default='seed0_models012'); a=p.parse_args(); torch.set_num_threads(1); start=time.perf_counter(); results={}
    for task in a.tasks.split(','):
        tr,te,dim,classes=make_data(task,a.n,a.data_seed); task_results={}
        for name,cfg in (('default',DEFAULT),('selected',SELECTED[task])):
            rows=[]
            for seed in [int(s) for s in a.model_seeds.split(',')]:
                best,final=run(dim,classes,tr,te,cfg,seed,a.epochs,a.lr,a.batch); rows.append({'seed':seed,'best_test_acc':best,'final_test_acc':final})
            vals=[r['best_test_acc'] for r in rows]; task_results[name]={'config':cfg,'runs':rows,'mean_best':sum(vals)/len(vals),'std_best':float(torch.tensor(vals).std(unbiased=False))}
            print(f"{task:18s} {name:9s} mean={task_results[name]['mean_best']:.4f} std={task_results[name]['std_best']:.4f} values={[round(v,4) for v in vals]}")
        results[task]=task_results
    out='local_electrical_nf_usage_results'; os.makedirs(out,exist_ok=True); result={'config':vars(a),'results':results,'seconds_total':time.perf_counter()-start}; path=os.path.join(out,'validate_field_hyperparams_'+a.result_tag+'.json');
    with open(path,'w',encoding='utf-8') as f: json.dump(result,f,indent=2)
    print('saved',path)


if __name__=='__main__': main()
