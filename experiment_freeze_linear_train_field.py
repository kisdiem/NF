"""Test whether Local NF can improve after the linear maps are frozen.

Procedure:
1. Train the complete Local NF for ``pretrain_epochs``.
2. Keep the best test-accuracy checkpoint as the operating point.
3. Freeze input/output Linear layers and train only ``field``.
4. Report whether field-only optimization improves that operating point.
"""
import argparse
import copy
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from local_electrical_nf_v3 import LocalElectricalNFV3MLP
from train_mnist import load_data


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total


def train_epoch(model, loader, optimizer, device):
    model.train()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pretrain-epochs', type=int, default=20)
    p.add_argument('--field-epochs', type=int, default=20)
    p.add_argument('--subset', type=int, default=5000)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cpu')
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--field-lr', type=float, default=3e-3)
    p.add_argument('--result-tag', default='seed0')
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    train_loader, test_loader = load_data('data/mnist', a.batch, a.subset)
    model = LocalElectricalNFV3MLP(8, 8, 4, mode='raw_bounded').to(device)
    all_optimizer = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)
    pretrain = []
    best_acc = -1.0
    best_state = None
    start = time.perf_counter()
    for epoch in range(a.pretrain_epochs):
        train_acc = train_epoch(model, train_loader, all_optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        pretrain.append({'epoch': epoch + 1, 'train_acc': train_acc, 'test_acc': test_acc})
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    best_epoch = max(pretrain, key=lambda r: r['test_acc'])['epoch']
    linear_before = {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith(('input.', 'output.'))}
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith('field.')
    field_optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=a.field_lr,
        weight_decay=1e-4,
    )
    field_only = []
    for epoch in range(a.field_epochs):
        train_acc = train_epoch(model, train_loader, field_optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        field_only.append({'epoch': epoch + 1, 'train_acc': train_acc, 'test_acc': test_acc})

    linear_delta = {}
    for n, before in linear_before.items():
        linear_delta[n] = float((model.state_dict()[n] - before).abs().max())
    field_delta = {}
    best_field_state = best_state
    for n, p in model.field.named_parameters():
        full_name = 'field.' + n
        before = best_state[full_name]
        field_delta[n] = {
            'abs_delta_mean': float((p.detach() - before).abs().mean()),
            'relative_delta': float((p.detach() - before).norm() / (before.norm() + 1e-8)),
        }
    result = {
        'config': vars(a),
        'model': 'LocalElectricalNFV3MLP(raw_bounded)',
        'pretrain_best_epoch': best_epoch,
        'pretrain_best_test_acc': best_acc,
        'pretrain_history': pretrain,
        'field_only_history': field_only,
        'field_only_best_test_acc': max(r['test_acc'] for r in field_only),
        'field_only_final_test_acc': field_only[-1]['test_acc'],
        'linear_parameter_max_delta_after_freeze': linear_delta,
        'field_parameter_delta_from_best_point': field_delta,
        'seconds_total': time.perf_counter() - start,
        'interpretation': 'Only field parameters are trainable after the best complete-model checkpoint.',
    }
    out = 'local_electrical_nf_usage_results'
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, 'freeze_linear_train_field_' + a.result_tag + '.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
