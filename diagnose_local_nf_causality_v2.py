"""Causal checks for whether Local Electrical NF contributes to prediction."""
import argparse, json, os
import torch
import torch.nn.functional as F
from local_electrical_nf_v3 import LocalElectricalNFV3MLP
from train_mnist import load_data


def evaluate(model, loader, device, transform="normal"):
    model.eval(); correct = total = 0
    old_steps = model.field.steps
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            h0 = model.input(x.flatten(1)).view(-1, 1, 8, 8)
            if transform == "pre_field": h = h0
            else:
                model.field.steps = 1 if transform == "one_step" else old_steps
                h = model.field(h0)
                if transform == "shuffle_space":
                    h = h.flatten(1)[:, torch.randperm(64, device=device)].view(-1, 1, 8, 8)
                elif transform == "spatial_mean":
                    h = h.mean(dim=(2, 3), keepdim=True).expand_as(h)
                elif transform == "zero_field": h = torch.zeros_like(h)
            z = model.output(h.flatten(1)); correct += (z.argmax(1) == y).sum().item(); total += y.numel()
    model.field.steps = old_steps
    return correct / total


def train(freeze, args, train_loader, test_loader, device):
    torch.manual_seed(0)
    model = LocalElectricalNFV3MLP(8, 8, 4, mode="raw_bounded").to(device)
    initial = {n: p.detach().clone() for n, p in model.field.named_parameters()}
    if freeze:
        for p in model.field.parameters(): p.requires_grad = False
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-3, weight_decay=1e-4)
    history = []
    for epoch in range(args.epochs):
        model.train(); correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device); z = model(x); loss = F.cross_entropy(z, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); correct += (z.argmax(1) == y).sum().item(); total += y.numel()
        history.append({"epoch": epoch + 1, "train_acc": correct / total, "test_acc": evaluate(model, test_loader, device)})
    delta = {}
    for n, p in model.field.named_parameters():
        delta[n] = {"abs_delta_mean": float((p.detach() - initial[n]).abs().mean()), "relative_delta": float((p.detach() - initial[n]).norm() / (initial[n].norm() + 1e-8))}
    result = {"freeze_field": freeze, "history": history, "field_parameter_delta": delta}
    if not freeze:
        result["post_training_perturbation"] = {k: evaluate(model, test_loader, device, k) for k in ("normal", "shuffle_space", "spatial_mean", "zero_field", "pre_field", "one_step")}
        result["field_gradient_check"] = {}
        x, y = next(iter(train_loader)); x, y = x.to(device), y.to(device); model.zero_grad(set_to_none=True); F.cross_entropy(model(x), y).backward()
        for n, p in model.field.named_parameters(): result["field_gradient_check"][n] = float(p.grad.abs().mean()) if p.grad is not None else None
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--epochs", type=int, default=5); p.add_argument("--subset", type=int, default=5000); p.add_argument("--batch", type=int, default=128); p.add_argument("--device", default="cpu"); args = p.parse_args()
    device = torch.device(args.device); train_loader, test_loader = load_data("data/mnist", args.batch, args.subset)
    results = [train(False, args, train_loader, test_loader, device), train(True, args, train_loader, test_loader, device)]
    out = "local_electrical_nf_usage_results"; os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "causality_v2_seed0.json"), "w", encoding="utf-8") as f: json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
