"""Causal checks for whether the Local Electrical field matters."""
import argparse, json, os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from local_electrical_nf_v3 import LocalElectricalNFV3MLP


def load_data(root, batch, subset):
    tf = transforms.ToTensor()
    tr = datasets.MNIST(root, train=True, download=True, transform=tf)
    te = datasets.MNIST(root, train=False, download=True, transform=tf)
    if subset:
        tr = Subset(tr, range(subset))
    return DataLoader(tr, batch_size=batch, shuffle=True), DataLoader(te, batch_size=1024)


def evaluate(model, loader, device, transform="normal"):
    model.eval(); correct = total = 0; old_steps = model.field.steps
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            h0 = model.input(x.flatten(1)).view(-1, 1, 8, 8)
            if transform == "pre_field":
                h = h0
            else:
                model.field.steps = 1 if transform == "one_step" else old_steps
                h = model.field(h0)
                if transform == "shuffle_space":
                    perm = torch.randperm(64, device=device)
                    h = h.flatten(1)[:, perm].view(-1, 1, 8, 8)
                elif transform == "spatial_mean":
                    h = h.mean((2, 3), keepdim=True).expand_as(h)
                elif transform == "zero_field":
                    h = torch.zeros_like(h)
            z = model.output(h.flatten(1)); correct += (z.argmax(1) == y).sum().item(); total += y.numel()
    model.field.steps = old_steps
    return correct / total


def train(freeze, args, train_loader, test_loader, device):
    torch.manual_seed(args.seed)
    model = LocalElectricalNFV3MLP(8, 8, 4, mode="raw_bounded").to(device)
    initial = {n: p.detach().clone() for n, p in model.field.named_parameters()}
    if freeze:
        for p in model.field.parameters(): p.requires_grad = False
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    hist = []
    for epoch in range(args.epochs):
        model.train(); correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device); loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            with torch.no_grad(): correct += (model(x).argmax(1) == y).sum().item(); total += y.numel()
        hist.append({"epoch": epoch + 1, "train_acc": correct / total, "test_acc": evaluate(model, test_loader, device)})
    delta = {n: {"abs_delta_mean": float((p.detach()-initial[n]).abs().mean()), "relative_delta": float((p.detach()-initial[n]).norm()/(initial[n].norm()+1e-8))} for n,p in model.field.named_parameters()}
    result = {"freeze_field": freeze, "history": hist, "field_parameter_delta": delta}
    if not freeze:
        result["post_training_perturbation"] = {k: evaluate(model, test_loader, device, k) for k in ("normal","shuffle_space","spatial_mean","zero_field","pre_field","one_step")}
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--epochs",type=int,default=5); p.add_argument("--subset",type=int,default=5000); p.add_argument("--batch",type=int,default=128); p.add_argument("--lr",type=float,default=3e-3); p.add_argument("--seed",type=int,default=0); p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu"); p.add_argument("--data-root",default="data/mnist"); p.add_argument("--result",default="results/field_causality_new.json"); a=p.parse_args()
    device=torch.device(a.device); tr,te=load_data(a.data_root,a.batch,a.subset); results=[train(False,a,tr,te,device),train(True,a,tr,te,device)]; os.makedirs(os.path.dirname(a.result) or ".",exist_ok=True); json.dump(results,open(a.result,"w",encoding="utf-8"),indent=2); print(json.dumps(results,indent=2))
if __name__ == "__main__": main()
