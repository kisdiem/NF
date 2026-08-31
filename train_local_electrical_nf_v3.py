"""Screen stability fixes for Local Electrical NF without overwriting v1/v2."""
import argparse, json, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from local_electrical_nf import LocalElectricalNFMLP
from local_electrical_nf_v3 import LocalElectricalFieldV3, LocalElectricalNFV3MLP
from train_mnist import load_data

class MLP(nn.Module):
    def __init__(self, act):
        super().__init__(); self.a=nn.Linear(784,64); self.b=nn.Linear(64,10); self.act=act
    def forward(self,x):
        h=self.a(x.flatten(1)); return self.b(self.act(h))

def make_model(kind,args):
    if kind=="relu": return MLP(F.relu)
    if kind=="gelu": return MLP(F.gelu)
    if kind=="no_inhibition":
        return LocalElectricalNFMLP(8,8,args.steps,threshold_init=.5,strength_init=.5,decay_init=.8,tau=.2,no_threshold=False,persistence=True,inhibition=False)
    steps=1 if kind=="one_step_normalized" else args.steps
    return LocalElectricalNFV3MLP(8,8,steps,mode=kind)

def save_diag(field,out_dir,name):
    with open(os.path.join(out_dir,name+"_diagnostics.json"),"w",encoding="utf-8") as f: json.dump(field.last_diagnostics,f,indent=2,ensure_ascii=False)
    try:
        import matplotlib.pyplot as plt
        for t,s in enumerate(field.last_states):
            for data,suf,title in ((s,"membrane","membrane"),(field.last_inhibitions[t],"inhibition","inhibition")):
                plt.figure(figsize=(3,3)); plt.imshow(data[0,0].cpu(),cmap="magma"); plt.colorbar(); plt.title(f"{name} {title} t={t}"); plt.tight_layout(); plt.savefig(os.path.join(out_dir,f"{name}_{suf}_t{t}.png"),dpi=130); plt.close()
        d=field.last_diagnostics; plt.figure(figsize=(5,3)); plt.plot(d["state_change"],marker="o",label="state change"); plt.plot(d["activation_rate"],marker="s",label="activation rate"); plt.plot(d["inhibition_gate_mean"],marker="^",label="inhibition gate"); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out_dir,name+"_diagnostics.png"),dpi=130); plt.close()
    except Exception as exc: print("visualization skipped:",exc)

def run(kind,args,tr,te,device,out_dir):
    torch.manual_seed(args.seed); model=make_model(kind,args).to(device); opt=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-4); hist=[]; start=time.process_time()
    for ep in range(args.epochs):
        model.train(); ls=correct=total=0
        for x,y in tr:
            x,y=x.to(device),y.to(device); logits=model(x); loss=F.cross_entropy(logits,y); opt.zero_grad(set_to_none=True); loss.backward(); gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); ls+=loss.item()*y.numel(); correct+=(logits.argmax(1)==y).sum().item(); total+=y.numel()
        model.eval(); tl=tc=tt=0
        with torch.no_grad():
            for x,y in te:
                x,y=x.to(device),y.to(device); z=model(x); tl+=F.cross_entropy(z,y).item()*y.numel(); tc+=(z.argmax(1)==y).sum().item(); tt+=y.numel()
        row={"epoch":ep+1,"train_loss":ls/total,"train_acc":correct/total,"test_loss":tl/tt,"test_acc":tc/tt,"grad_norm":float(gn)}; hist.append(row); print(f"{kind:24s} ep{ep+1:02d}/{args.epochs} train={row['train_acc']:.4f} test={row['test_acc']:.4f}")
    result={"model":kind,"history":hist,"best_test_acc":max(r["test_acc"] for r in hist),"final_test_acc":hist[-1]["test_acc"],"parameters":sum(p.numel() for p in model.parameters()),"cpu_seconds":time.process_time()-start}
    if hasattr(model,"field"): result["diagnostics"]=model.field.last_diagnostics; result["field_report"]=model.field.parameter_report(); save_diag(model.field,out_dir,kind)
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument("--epochs",type=int,default=20); p.add_argument("--subset",type=int,default=5000); p.add_argument("--batch",type=int,default=128); p.add_argument("--steps",type=int,default=4); p.add_argument("--lr",type=float,default=3e-3); p.add_argument("--seed",type=int,default=0); p.add_argument("--models",default="relu,gelu,no_inhibition,normalized,bounded,homeostatic,refractory,refractory_bounded,one_step_normalized"); p.add_argument("--result-tag",default="seed0"); p.add_argument("--data-root",default="data/mnist"); p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu"); a=p.parse_args(); device=torch.device(a.device); out="local_electrical_nf_v3_results"; os.makedirs(out,exist_ok=True); tr,te=load_data(a.data_root,a.batch,a.subset); results=[run(k,a,tr,te,device,out) for k in a.models.split(",")]; path=os.path.join(out,"results_"+a.result_tag+".json"); json.dump(results,open(path,"w",encoding="utf-8"),indent=2,ensure_ascii=False); print("\nSUMMARY"); [print(f"{r['model']:24s} best={r['best_test_acc']:.4f} final={r['final_test_acc']:.4f} params={r['parameters']} time={r['cpu_seconds']:.2f}s") for r in results]; print("artifacts:",os.path.abspath(out))
if __name__=="__main__": main()
