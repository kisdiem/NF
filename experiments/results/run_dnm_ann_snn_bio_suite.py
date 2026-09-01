"""Compare compact ANN, multiplicative DNM, LIF-SNN and Bio-easy."""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from models.bio_neuron import BioMLP
from experiments.results.run_bio_easy_old_logic_suite import data

TASKS = "xor,circles,moons,checkerboard,parity8,noisy_moons100,noisy_spiral100,hierarchical_xor16,multiplexer,cnf_dnf,carry8,variable_parity16,equality16,majority32,delayed_xor20,fsm_sequence"

class ANN(nn.Module):
    def __init__(self,d,h,c): super().__init__(); self.net=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Linear(h,c))
    def forward(self,x): return self.net(x.flatten(1))

class DNM(nn.Module):
    """Classic compact dendritic baseline: branch nonlinearities fused multiplicatively."""
    def __init__(self,d,h,c,b=4):
        super().__init__(); self.proj=nn.Linear(d,h); self.w=nn.Parameter(torch.randn(h,b,h)*0.08); self.bias=nn.Parameter(torch.zeros(h,b)); self.out=nn.Linear(h,c); self.b=b
    def forward(self,x):
        h=torch.tanh(self.proj(x.flatten(1))); z=torch.einsum('nk,ibk->nib',h,self.w)+self.bias.unsqueeze(0)
        branch=torch.tanh(z); fused=torch.prod(1.0+0.5*branch,dim=2)-1.0; return self.out(fused)

class SNN(nn.Module):
    def __init__(self,d,h,c,steps=5,decay=.8):
        super().__init__(); self.proj=nn.Linear(d,h); self.out=nn.Linear(h,c); self.theta=nn.Parameter(torch.full((h,),.5)); self.steps=steps; self.decay=decay
    def forward(self,x):
        cur=self.proj(x.flatten(1)); v=torch.zeros_like(cur); spikes=[]
        for _ in range(self.steps):
            v=self.decay*v+cur
            spikes.append(torch.sigmoid(4.0*(v-self.theta)))
        return self.out(torch.stack(spikes).mean(0))

def make(kind,d,c):
    if kind=='ann': return ANN(d,64,c)
    if kind=='dnm': return DNM(d,64,c)
    if kind=='snn': return SNN(d,64,c)
    return BioMLP(d,64,c,branches=4,steps=1,temporal=False,membrane_decay=0.,dendrite='soft_threshold',inhibition=True,adaptive_threshold=True,output_mode='mean')

def run(task,kind,args):
    torch.manual_seed(args.seed); xtr,xte,ytr,yte,c=data(task,args.n,args.seed); m=make(kind,xtr.shape[1],c).to(args.device); p=sum(q.numel() for q in m.parameters())
    opt=torch.optim.AdamW(m.parameters(),lr=args.lr,weight_decay=1e-4); tr=DataLoader(TensorDataset(xtr,ytr),args.batch,shuffle=True); te=DataLoader(TensorDataset(xte,yte),1024); hist=[]; start=time.perf_counter()
    for ep in range(1,args.epochs+1):
        m.train(); good=total=loss_sum=0.
        for xb,yb in tr:
            xb,yb=xb.to(args.device),yb.to(args.device); z=m(xb); loss=F.cross_entropy(z,yb); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5.); opt.step(); loss_sum+=loss.item()*yb.numel(); good+=(z.argmax(1)==yb).sum().item(); total+=yb.numel()
        m.eval(); gt=nt=0
        with torch.no_grad():
            for xb,yb in te:
                z=m(xb.to(args.device)); gt+=(z.argmax(1)==yb.to(args.device)).sum().item(); nt+=yb.numel()
        row={'epoch':ep,'train_loss':loss_sum/total,'train_acc':good/total,'test_acc':gt/nt}; hist.append(row)
        if ep==1 or ep%10==0 or ep==args.epochs: print(f'[{kind} {task}] ep={ep:03d} train={row["train_acc"]:.4f} test={row["test_acc"]:.4f}',flush=True)
    sec=time.perf_counter()-start; return {'task':task,'model':kind,'seed':args.seed,'parameters':p,'seconds':sec,'best_test_acc':max(x['test_acc'] for x in hist),'final_test_acc':hist[-1]['test_acc'],'history':hist}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--tasks',default=TASKS); p.add_argument('--models',default='ann,dnm,snn,bio-easy'); p.add_argument('--epochs',type=int,default=100); p.add_argument('--n',type=int,default=6000); p.add_argument('--batch',type=int,default=128); p.add_argument('--lr',type=float,default=3e-3); p.add_argument('--seed',type=int,default=0); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); p.add_argument('--result',required=True); a=p.parse_args(); rows=[]; os.makedirs(os.path.dirname(os.path.abspath(a.result)),exist_ok=True)
    for t in a.tasks.split(','):
        for k in a.models.split(','):
            rows.append(run(t,k,a)); json.dump({'config':vars(a),'results':rows},open(a.result,'w',encoding='utf-8'),indent=2,ensure_ascii=False)
    print('SUMMARY')
    for t in a.tasks.split(','):
        rr=[r for r in rows if r['task']==t]; print(t,[(r['model'],round(r['best_test_acc']*100,2),r['parameters']) for r in rr])
if __name__=='__main__': main()
