"""Compare Transformer, MLP, bounded Local NF, and unrestricted Local NF."""
import argparse, json, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from benchmark_hard_tasks import make_data, MLP, LocalNF


class FeatureTransformer(nn.Module):
    def __init__(self, dim, classes, d_model=16, layers=2, heads=2):
        super().__init__(); self.embed=nn.Linear(1,d_model); self.cls=nn.Parameter(torch.zeros(1,1,d_model)); self.pos=nn.Parameter(torch.zeros(1,dim+1,d_model))
        block=nn.TransformerEncoderLayer(d_model=d_model,nhead=heads,dim_feedforward=32,dropout=0.0,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(block,num_layers=layers,enable_nested_tensor=False); self.out=nn.Linear(d_model,classes); nn.init.normal_(self.pos,std=.02)
    def forward(self,x):
        h=self.embed(x.unsqueeze(-1)); cls=self.cls.expand(x.shape[0],-1,-1); h=torch.cat((cls,h),1)+self.pos[:,:x.shape[1]+1]; return self.out(self.encoder(h)[:,0])


def evaluate(m,loader):
    m.eval(); c=t=0
    with torch.inference_mode():
        for x,y in loader: c+=(m(x).argmax(1)==y).sum().item(); t+=y.numel()
    return c/t


def run(kind,tr,te,dim,classes,a):
    torch.manual_seed(a.seed)
    if kind=='relu': m=MLP(dim,classes,F.relu)
    elif kind=='gelu': m=MLP(dim,classes,F.gelu)
    elif kind in ('raw_bounded','raw_unbounded'):
        m=LocalNF(dim,classes,kind)
    else: m=FeatureTransformer(dim,classes,a.d_model,a.layers,a.heads)
    train=DataLoader(tr,batch_size=a.batch,shuffle=True); test=DataLoader(te,batch_size=1024); opt=torch.optim.Adam(m.parameters(),lr=a.lr,weight_decay=1e-4); hist=[]; st=time.perf_counter()
    for ep in range(a.epochs):
        m.train(); c=t=0
        for x,y in train:
            z=m(x); loss=F.cross_entropy(z,y); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step(); c+=(z.argmax(1)==y).sum().item(); t+=y.numel()
        hist.append({'epoch':ep+1,'train_acc':c/t,'test_acc':evaluate(m,test)})
    return {'model':kind,'best_test_acc':max(r['test_acc'] for r in hist),'final_test_acc':hist[-1]['test_acc'],'history':hist,'parameters':sum(p.numel() for p in m.parameters()),'seconds':time.perf_counter()-st}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--tasks',default='spiral3,checkerboard,parity8,noisy_moons100,noisy_spiral100'); p.add_argument('--n',type=int,default=6000); p.add_argument('--epochs',type=int,default=100); p.add_argument('--batch',type=int,default=128); p.add_argument('--lr',type=float,default=3e-3); p.add_argument('--seed',type=int,default=0); p.add_argument('--d-model',type=int,default=16); p.add_argument('--layers',type=int,default=2); p.add_argument('--heads',type=int,default=2); p.add_argument('--result-tag',default='seed0'); a=p.parse_args(); torch.set_num_threads(1); out='transformer_task_results'; os.makedirs(out,exist_ok=True); allr={}
    for task in a.tasks.split(','):
        tr,te,dim,classes=make_data(task,a.n,a.seed); allr[task]=[run(k,tr,te,dim,classes,a) for k in ('relu','gelu','raw_bounded','raw_unbounded','transformer')]; print('\n'+task); [print(f"{r['model']:14s} best={r['best_test_acc']:.4f} final={r['final_test_acc']:.4f} params={r['parameters']} time={r['seconds']:.2f}s") for r in allr[task]]
    with open(os.path.join(out,'results_'+a.result_tag+'.json'),'w',encoding='utf-8') as f: json.dump(allr,f,indent=2)
if __name__=='__main__': main()
