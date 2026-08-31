"""Systematic ablations of what can make the Local Electrical Field useful.

The complete model is first trained to a diagnostic best point.  For every
variant, input/output linear maps are copied from that point and frozen; only
the selected field parameters/mechanism is trained.  This isolates the field
contribution and avoids comparing different learned linear representations.
"""
import argparse, copy, json, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from local_electrical_nf_v3 import LocalElectricalFieldV3, LocalElectricalNFV3MLP
from train_mnist import load_data


def accuracy(model, loader, device):
    model.eval(); correct = total = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item(); total += y.numel()
    return correct / total


def complete_pretrain(model, loader, test_loader, epochs, lr, device):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    history=[]; best=-1.; best_state=None
    for ep in range(epochs):
        model.train(); correct=total=0
        for x,y in loader:
            x,y=x.to(device),y.to(device); z=model(x); loss=F.cross_entropy(z,y)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
            correct+=(z.argmax(1)==y).sum().item(); total+=y.numel()
        test=accuracy(model,test_loader,device); row={'epoch':ep+1,'train_acc':correct/total,'test_acc':test}; history.append(row)
        if test>best: best=test; best_state=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return best, history, best_state


class FrozenLinearModel(torch.nn.Module):
    def __init__(self, source, field):
        super().__init__(); self.input=copy.deepcopy(source.input); self.field=field; self.output=copy.deepcopy(source.output)
        for p in self.input.parameters(): p.requires_grad=False
        for p in self.output.parameters(): p.requires_grad=False
    def forward(self,x): return self.output(self.field(self.input(x.flatten(1)).view(-1,1,8,8)).flatten(1))


def make_field(kind, source_state):
    mode='raw_bounded'; steps=4
    if kind == 'no_inhibition': mode='raw_unbounded'
    if kind == 'one_step': steps=1
    if kind == 'two_steps': steps=2
    if kind == 'eight_steps': steps=8
    field=LocalElectricalFieldV3(8,8,steps,mode=mode,collect_diagnostics=False)
    field.load_state_dict(source_state, strict=False)
    if kind == 'kernel_only':
        field.exc_kernel = nn.Parameter(field.exc_kernel.detach().clone())
    if kind == 'no_threshold':
        with torch.no_grad(): field.theta_raw.fill_(-20.)
    if kind == 'no_persistence':
        with torch.no_grad(): field.decay_raw.fill_(-20.)
    return field


def train_field(model, kind, loader, test_loader, epochs, lr, device):
    # Every experiment trains only a specified group of field parameters.
    for name,p in model.field.named_parameters():
        if kind == 'frozen': train=False
        elif kind == 'all' or kind.startswith('all_lr'): train=True
        elif kind == 'neuron_params': train=name in ('theta_raw','strength_raw','decay_raw')
        elif kind == 'threshold_only': train=name=='theta_raw'
        elif kind == 'strength_only': train=name=='strength_raw'
        elif kind == 'decay_only': train=name=='decay_raw'
        elif kind == 'inhibition_only': train=name in ('rho_raw','beta_raw')
        elif kind == 'kernel_only': train=name=='exc_kernel'
        elif kind == 'no_threshold': train=name != 'theta_raw'
        elif kind == 'no_persistence': train=name != 'decay_raw'
        else: train=True
        p.requires_grad=train
    params=[p for p in model.field.parameters() if p.requires_grad]
    opt=torch.optim.Adam(params,lr=lr,weight_decay=1e-4) if params else None
    history=[]
    initial={n:p.detach().clone() for n,p in model.field.named_parameters()}
    for ep in range(epochs):
        model.train(); correct=total=0
        for x,y in loader:
            x,y=x.to(device),y.to(device); z=model(x); loss=F.cross_entropy(z,y)
            if opt is not None:
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.); opt.step()
            correct+=(z.argmax(1)==y).sum().item(); total+=y.numel()
        history.append({'epoch':ep+1,'train_acc':correct/total,'test_acc':accuracy(model,test_loader,device)})
    delta={n:float((p.detach()-initial[n]).abs().mean()) for n,p in model.field.named_parameters()}
    return history, delta


def main():
    p=argparse.ArgumentParser(); p.add_argument('--pretrain-epochs',type=int,default=20); p.add_argument('--field-epochs',type=int,default=20); p.add_argument('--subset',type=int,default=5000); p.add_argument('--batch',type=int,default=128); p.add_argument('--lr',type=float,default=3e-3); p.add_argument('--seed',type=int,default=0); p.add_argument('--device',default='cpu'); p.add_argument('--result-tag',default='seed0_20plus20'); p.add_argument('--only',default=''); a=p.parse_args()
    torch.set_num_threads(1); torch.manual_seed(a.seed); device=torch.device(a.device)
    loader,test_loader=load_data('data/mnist',a.batch,a.subset)
    base=LocalElectricalNFV3MLP(8,8,4,mode='raw_bounded').to(device)
    best,pretrain,best_state=complete_pretrain(base,loader,test_loader,a.pretrain_epochs,a.lr,device)
    variants=[
        ('frozen',1.0),('all_lr0.1',a.lr*.1),('all_lr1',a.lr),('all_lr3',a.lr*3),('all_lr10',a.lr*10),
        ('neuron_params',a.lr),('threshold_only',a.lr),('strength_only',a.lr),('decay_only',a.lr),
        ('inhibition_only',a.lr),('kernel_only',a.lr),('no_inhibition',a.lr),('one_step',a.lr),
        ('two_steps',a.lr),('eight_steps',a.lr),('no_threshold',a.lr),('no_persistence',a.lr),
    ]
    if a.only:
        variants=[v for v in variants if v[0] in a.only.split(',')]
    results={}; start=time.perf_counter()
    for kind,vlr in variants:
        field=make_field(kind,base.field.state_dict()).to(device)
        model=FrozenLinearModel(base,field).to(device)
        history,delta=train_field(model,kind,loader,test_loader,a.field_epochs,vlr,device)
        results[kind]={'best_test_acc':max(r['test_acc'] for r in history),'final_test_acc':history[-1]['test_acc'],'parameters_total':sum(p.numel() for p in model.parameters()),'trainable_field_parameters':sum(p.numel() for p in field.parameters() if p.requires_grad),'history':history,'field_abs_delta_mean':delta}
        print(f"{kind:18s} best={results[kind]['best_test_acc']:.4f} final={results[kind]['final_test_acc']:.4f} trainable={results[kind]['trainable_field_parameters']}")
    result={'config':vars(a),'pretrain_best_test_acc':best,'pretrain_history':pretrain,'variants':results,'seconds_total':time.perf_counter()-start}
    out='local_electrical_nf_usage_results'; os.makedirs(out,exist_ok=True); path=os.path.join(out,'field_all_ablation_'+a.result_tag+'.json')
    with open(path,'w',encoding='utf-8') as f: json.dump(result,f,indent=2)
    print('saved',path)


if __name__=='__main__': main()
