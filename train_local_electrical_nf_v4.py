"""Screen energy-scale/stability trade-offs for Local Electrical NF."""
import argparse,json,os,time
import torch
import torch.nn as nn
import torch.nn.functional as F
from local_electrical_nf import LocalElectricalNFMLP
from local_electrical_nf_v3 import LocalElectricalNFV3MLP
from train_mnist import load_data

class MLP(nn.Module):
    def __init__(self,a): super().__init__(); self.a=nn.Linear(784,64); self.b=nn.Linear(64,10); self.act=a
    def forward(self,x): return self.b(self.act(self.a(x.flatten(1))))

def make_model(k,a):
    if k=='relu': return MLP(F.relu)
    if k=='gelu': return MLP(F.gelu)
    if k=='no_inhibition': return LocalElectricalNFMLP(8,8,a.steps,threshold_init=.5,strength_init=.5,decay_init=.8,tau=.2,no_threshold=False,persistence=True,inhibition=False)
    cfg={'normalized_balanced':dict(mode='normalized',excitation_gain=4.0,beta_init=3.0,decay_init=.7),
         'strong_inhibition':dict(mode='normalized',excitation_gain=4.0,beta_init=5.0,rho_init=.1,decay_init=.7),
         'leaky_balanced':dict(mode='normalized',excitation_gain=4.0,beta_init=5.0,rho_init=.1,decay_init=.45),
         'full_scale_balanced':dict(mode='normalized',excitation_gain=6.8,beta_init=6.0,rho_init=.15,decay_init=.6),
         'bounded_balanced':dict(mode='bounded',excitation_gain=4.0,beta_init=5.0,rho_init=.1,decay_init=.7),
         'refractory_balanced':dict(mode='refractory_bounded',excitation_gain=4.0,beta_init=5.0,rho_init=.1,decay_init=.7,gamma_init=.5),
         'centered_full':dict(mode='centered',excitation_gain=6.8,beta_init=6.0,rho_init=.15,decay_init=.6),
         'centered_refractory':dict(mode='centered_refractory',excitation_gain=6.8,beta_init=6.0,rho_init=.15,decay_init=.6,gamma_init=.5),
         'soft_reset':dict(mode='soft_reset',excitation_gain=6.8,beta_init=6.0,rho_init=.15,decay_init=.6),
         'raw_bounded':dict(mode='raw_bounded',excitation_gain=1.0,beta_init=1.0,rho_init=.15,decay_init=.8),
         'raw_strong_bounded':dict(mode='raw_bounded',excitation_gain=1.0,beta_init=4.0,rho_init=.1,decay_init=.7)}[k]
    return LocalElectricalNFV3MLP(8,8,a.steps,**cfg)

def run(k,a,tr,te,dev,out):
    torch.manual_seed(a.seed); m=make_model(k,a).to(dev); opt=torch.optim.Adam(m.parameters(),lr=a.lr,weight_decay=1e-4); hist=[]; st=time.process_time()
    for ep in range(a.epochs):
        m.train(); ls=co=tot=0
        for x,y in tr:
            x,y=x.to(dev),y.to(dev); z=m(x); loss=F.cross_entropy(z,y); opt.zero_grad(set_to_none=True); loss.backward(); gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step(); ls+=loss.item()*y.numel(); co+=(z.argmax(1)==y).sum().item(); tot+=y.numel()
        m.eval(); tl=tc=tt=0
        with torch.no_grad():
            for x,y in te:
                x,y=x.to(dev),y.to(dev); z=m(x); tl+=F.cross_entropy(z,y).item()*y.numel(); tc+=(z.argmax(1)==y).sum().item(); tt+=y.numel()
        row={'epoch':ep+1,'train_loss':ls/tot,'train_acc':co/tot,'test_loss':tl/tt,'test_acc':tc/tt,'grad_norm':float(gn)}; hist.append(row); print(f'{k:24s} ep{ep+1:02d}/{a.epochs} train={row["train_acc"]:.4f} test={row["test_acc"]:.4f}')
    r={'model':k,'history':hist,'best_test_acc':max(x['test_acc'] for x in hist),'final_test_acc':hist[-1]['test_acc'],'parameters':sum(p.numel() for p in m.parameters()),'cpu_seconds':time.process_time()-st}
    if hasattr(m,'field'): r['diagnostics']=m.field.last_diagnostics; r['field_report']=m.field.parameter_report()
    return r

def main():
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=10); p.add_argument('--subset',type=int,default=5000); p.add_argument('--batch',type=int,default=128); p.add_argument('--steps',type=int,default=4); p.add_argument('--lr',type=float,default=3e-3); p.add_argument('--seed',type=int,default=0); p.add_argument('--models',default='relu,gelu,no_inhibition,normalized_balanced,strong_inhibition,leaky_balanced,full_scale_balanced,bounded_balanced,refractory_balanced'); p.add_argument('--result-tag',default='seed0'); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); p.add_argument('--data-root',default='data/mnist'); a=p.parse_args(); dev=torch.device(a.device); out='local_electrical_nf_v4_results'; os.makedirs(out,exist_ok=True); tr,te=load_data(a.data_root,a.batch,a.subset); rs=[run(k,a,tr,te,dev,out) for k in a.models.split(',')]; json.dump(rs,open(os.path.join(out,'results_'+a.result_tag+'.json'),'w',encoding='utf-8'),indent=2,ensure_ascii=False); print('\nSUMMARY'); [print(f"{r['model']:24s} best={r['best_test_acc']:.4f} final={r['final_test_acc']:.4f} params={r['parameters']} time={r['cpu_seconds']:.2f}s") for r in rs]
if __name__=='__main__': main()
