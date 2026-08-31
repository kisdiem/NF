"""Test whether field learning helps on hard datasets with remaining headroom."""
import argparse, copy, json, os, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from benchmark_hard_tasks import make_data, LocalNF


def accuracy(model, loader):
    model.eval(); correct=total=0
    with torch.inference_mode():
        for x,y in loader:
            correct += (model(x).argmax(1)==y).sum().item(); total += y.numel()
    return correct/total


def train(model, loader, test_loader, epochs, lr, freeze_field=False, freeze_linear=False):
    if freeze_field:
        for p in model.field.parameters(): p.requires_grad=False
    if freeze_linear:
        for p in model.a.parameters(): p.requires_grad=False
        for p in model.b.parameters(): p.requires_grad=False
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.Adam(params,lr=lr,weight_decay=1e-4)
    history=[]; best=-1.; best_state=None
    for ep in range(epochs):
        model.train(); correct=total=0
        for x,y in loader:
            z=model(x); loss=F.cross_entropy(z,y); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.); opt.step()
            correct+=(z.argmax(1)==y).sum().item(); total+=y.numel()
        test=accuracy(model,test_loader); row={'epoch':ep+1,'train_acc':correct/total,'test_acc':test}; history.append(row)
        if test>best: best=test; best_state=copy.deepcopy(model.state_dict())
    return best, history, best_state


def main():
    p=argparse.ArgumentParser(); p.add_argument('--tasks',default='checkerboard,noisy_moons100,noisy_spiral100'); p.add_argument('--n',type=int,default=6000); p.add_argument('--epochs',type=int,default=100); p.add_argument('--field-epochs',type=int,default=30); p.add_argument('--batch',type=int,default=128); p.add_argument('--lr',type=float,default=3e-3); p.add_argument('--seed',type=int,default=0); p.add_argument('--result-tag',default='seed0'); a=p.parse_args(); torch.set_num_threads(1); torch.manual_seed(a.seed)
    all_results={}; start=time.perf_counter()
    for task in a.tasks.split(','):
        tr,te,dim,classes=make_data(task,a.n,a.seed); train_loader=DataLoader(tr,batch_size=a.batch,shuffle=True); test_loader=DataLoader(te,batch_size=1024)
        torch.manual_seed(a.seed); full=LocalNF(dim,classes,'raw_bounded'); full_best,full_hist,full_state=train(full,train_loader,test_loader,a.epochs,a.lr)
        torch.manual_seed(a.seed); frozen=LocalNF(dim,classes,'raw_bounded'); frozen_best,frozen_hist,_=train(frozen,train_loader,test_loader,a.epochs,a.lr,freeze_field=True)
        field_only=LocalNF(dim,classes,'raw_bounded'); field_only.load_state_dict(full_state)
        field_best,field_hist,field_state=train(field_only,train_loader,test_loader,a.field_epochs,a.lr,freeze_linear=True)
        linear_delta={n:float((field_state[n]-full_state[n]).abs().max()) for n in full_state if n.startswith(('a.','b.'))}
        field_delta={n:float((field_state[n]-full_state[n]).abs().mean()) for n in full_state if n.startswith('field.')}
        all_results[task]={
            'full_train_best':full_best,'full_train_final':full_hist[-1]['test_acc'],
            'frozen_field_best':frozen_best,'frozen_field_final':frozen_hist[-1]['test_acc'],
            'field_only_from_full_best':field_best,'field_only_final':field_hist[-1]['test_acc'],
            'full_history':full_hist,'frozen_field_history':frozen_hist,'field_only_history':field_hist,
            'linear_max_delta_during_field_only':linear_delta,'field_abs_delta_from_full_best':field_delta,
        }
        print(f"{task:18s} full={full_best:.4f} frozen_field={frozen_best:.4f} field_only={field_best:.4f}")
    result={'config':vars(a),'tasks':all_results,'seconds_total':time.perf_counter()-start}
    out='local_electrical_nf_usage_results'; os.makedirs(out,exist_ok=True); path=os.path.join(out,'field_learning_hard_tasks_'+a.result_tag+'.json')
    with open(path,'w',encoding='utf-8') as f: json.dump(result,f,indent=2)
    print('saved',path)


if __name__=='__main__': main()
