"""Causal diagnostic: train Local NF with a trainable or frozen field.

The field remains in the forward path in both cases.  This distinguishes
"receives gradients" from "materially improves the learned representation".
"""
import argparse, json, os
import torch
import torch.nn.functional as F
from local_electrical_nf_v3 import LocalElectricalNFV3MLP
from train_mnist import load_data


def run(freeze, epochs, subset, batch, device):
    torch.manual_seed(0)
    train_loader, test_loader = load_data("data/mnist", batch, subset)
    model = LocalElectricalNFV3MLP(8, 8, 4, mode="raw_bounded").to(device)
    if freeze:
        for parameter in model.field.parameters():
            parameter.requires_grad = False
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-3, weight_decay=1e-4)
    history = []
    for epoch in range(epochs):
        model.train(); correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device); logits = model(x); loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            correct += (logits.argmax(1) == y).sum().item(); total += y.numel()
        model.eval(); test_correct = test_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                logits = model(x.to(device)); test_correct += (logits.argmax(1) == y.to(device)).sum().item(); test_total += y.numel()
        history.append({"epoch": epoch + 1, "train_acc": correct / total, "test_acc": test_correct / test_total})
    return {"freeze_field": freeze, "history": history, "final_test_acc": history[-1]["test_acc"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--epochs", type=int, default=5); parser.add_argument("--subset", type=int, default=5000); parser.add_argument("--batch", type=int, default=128); parser.add_argument("--device", default="cpu"); args = parser.parse_args()
    output = "local_electrical_nf_usage_results"; os.makedirs(output, exist_ok=True)
    results = [run(False, args.epochs, args.subset, args.batch, args.device), run(True, args.epochs, args.subset, args.batch, args.device)]
    with open(os.path.join(output, "freeze_field_seed0.json"), "w", encoding="utf-8") as f: json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
