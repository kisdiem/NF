"""Train Local NF neuron attributes separately on hard datasets.

Each variant starts from the same complete-model best checkpoint.  The input
and output Linear layers are frozen, so the experiment measures whether a
specific field attribute can improve an already established representation.
"""
import argparse, copy, json, os, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from benchmark_hard_tasks import make_data, LocalNF


def acc(model, loader):
    model.eval(); c=t=0
    with torch.inference_mode():
        for x,y in loader: c+=(model(x).argmax(1)==y).sum().item(); t+=y.numel()
    return c/t


def train(model, loader, test_loader, epochs, lr, train_names=None, field_only=True):
    for name,p in model.named_parameters():
        p.requires_grad = (name.startswith('field.') and (train_names is None or name.split('.',1)[1] in train_names)) if field_only else True
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.Adam(params,lr=lr,weight_decay=1e-4) if params else None
    history=[]; best=-1.; best_state=None
    for ep in range(epochs):
        model.train(); c=t=0
        for x,y in loader:
            z=model(x); loss=F.cross_entropy(z,y)
            if opt is not None:
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.); opt.step()
            c+=(z.argmax(1)==y).sum().item(); t+=y.numel()
        test=acc(model,test_loader); row={'epoch':ep+1,'train_acc':c/t,'test_acc':test}; history.append(row)
        if test>best: best=test; best_state=copy.deepcopy(model.state_dict())
    return best,history,best_state


def main():
    p=argparse.ArgumentParser(); p.add_argument('--tasks',default='checkerboard,noisy_moons100,noisy_spiral100'); p.add_argument('--n',type=int,default=6000); p.add_argument('--pretrain-epochs',type=int,default=100); p.add_argument('--field-epochs',type=int,default=30); p.add_argument('--batch',type=int,default=128); p.add_argument('--lr',type=float,default=3e-3); p.add_argument('--seed',type=int,default=0); p.add_argument('--result-tag',default='seed0_100plus30'); a=p.parse_args(); torch.set_num_threads(1); torch.manual_seed(a.seed); start=time.perf_counter(); all_results={}
    variants={
        'frozen':set(),
        'threshold_only':{'theta_raw'},
        'strength_only':{'strength_raw'},
        'decay_only':{'decay_raw'},
        'threshold_decay':{'theta_raw','decay_raw'},
        'neuron_all':{'theta_raw','strength_raw','decay_raw'},
        'global_inhibition':{'rho_raw','beta_raw'},
    }
    for task in a.tasks.split(','):
        tr,te,dim,classes=make_data(task,a.n,a.seed); loader=DataLoader(tr,a.batch,shuffle=True); test_loader=DataLoader(te,1024)
        torch.manual_seed(a.seed); base=LocalNF(dim,classes,'raw_bounded'); full_best,full_hist,full_state=train(base,loader,test_loader,a.pretrain_epochs,a.lr,field_only=False)
        rows={}
        for name,names in variants.items():
            model=LocalNF(dim,classes,'raw_bounded'); model.load_state_dict(full_state)
            initial={n:p.detach().clone() for n,p in model.field.named_parameters()}
            best,hist,best_state=train(model,loader,test_loader,a.field_epochs,a.lr,train_names=names)
            deltas={n:float((best_state['field.'+n]-initial[n]).abs().mean()) for n in initial if 'field.'+n in best_state}
            rows[name]={'best_test_acc':best,'final_test_acc':hist[-1]['test_acc'],'trainable_attributes':sorted(names),'history':hist,'field_abs_delta_mean':deltas}
            print(f"{task:18s} {name:18s} best={best:.4f} final={hist[-1]['test_acc']:.4f}")
        all_results[task]={'full_pretrain_best':full_best,'full_pretrain_history':full_hist,'variants':rows}
    out='local_electrical_nf_usage_results'; os.makedirs(out,exist_ok=True); result={'config':vars(a),'tasks':all_results,'seconds_total':time.perf_counter()-start}; path=os.path.join(out,'neuron_attribute_training_'+a.result_tag+'.json');
    with open(path,'w',encoding='utf-8') as f: json.dump(result,f,indent=2)
    print('saved',path)


if __name__=='__main__': main()
