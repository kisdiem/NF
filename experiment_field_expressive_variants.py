"""Field-only experiments that add expressive local computations.

All variants start from the same best complete Local NF checkpoint.  The
input/output Linear layers are then frozen, so improvements measure the
contribution of the field itself rather than a better linear representation.
"""
import argparse
import copy
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from local_electrical_nf_v3 import LocalElectricalFieldV3, LocalElectricalNFV3MLP
from train_mnist import load_data


class LearnableKernelField(LocalElectricalFieldV3):
    """Current local field with a trainable 3x3 excitation stencil."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, fuse_local_convs=False, collect_diagnostics=False)
        self.exc_kernel = nn.Parameter(self.exc_kernel.detach().clone())


class InteractionField(LocalElectricalFieldV3):
    """Adds an initially-zero local membrane/excitation product term."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, fuse_local_convs=False, collect_diagnostics=False)
        self.interaction_gain = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        theta, strength, decay, rho, beta, gamma = self._effective_parameters()
        v = x
        r_state = torch.zeros_like(v)
        for _ in range(self.steps):
            gate = torch.sigmoid((v - theta) / self.tau)
            release = gate * strength
            excitation = self.excitation_gain * F.conv2d(release, self.exc_kernel, padding=1)
            activity = F.conv2d(release, self.inh_kernel, padding=1)
            inh_gate = torch.sigmoid((activity - rho) / self.tau_inhibition)
            inhibition = beta * inh_gate * activity
            incoming = excitation - inhibition
            if self.mode in ('bounded', 'refractory_bounded', 'raw_bounded'):
                incoming = torch.tanh(incoming)
            # This term is local and starts exactly at zero.
            incoming = incoming + self.interaction_gain * torch.tanh(v * excitation)
            v = decay * v + incoming
        return v


class GatedField(LocalElectricalFieldV3):
    """Makes local propagation depend on the receiving membrane potential."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, fuse_local_convs=False, collect_diagnostics=False)
        self.gate_slope = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        theta, strength, decay, rho, beta, gamma = self._effective_parameters()
        v = x
        r_state = torch.zeros_like(v)
        for _ in range(self.steps):
            gate = torch.sigmoid((v - theta) / self.tau)
            release = gate * strength
            excitation = self.excitation_gain * F.conv2d(release, self.exc_kernel, padding=1)
            activity = F.conv2d(release, self.inh_kernel, padding=1)
            inh_gate = torch.sigmoid((activity - rho) / self.tau_inhibition)
            inhibition = beta * inh_gate * activity
            incoming = excitation - inhibition
            if self.mode in ('bounded', 'refractory_bounded', 'raw_bounded'):
                incoming = torch.tanh(incoming)
            # At slope=0 this is exactly the baseline; training can learn
            # different amplification/suppression for different V values.
            local_gate = torch.sigmoid(self.gate_slope * v)
            v = decay * v + incoming * (0.5 + local_gate)
        return v


class FieldOnlyModel(nn.Module):
    def __init__(self, input_layer, output_layer, field):
        super().__init__()
        self.input = copy.deepcopy(input_layer)
        self.field = field
        self.output = copy.deepcopy(output_layer)

    def forward(self, x):
        h = self.input(x.flatten(1)).view(-1, 1, 8, 8)
        return self.output(self.field(h).flatten(1))


def accuracy(model, loader, device):
    model.eval(); correct = total = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total


def train_one(model, loader, test_loader, epochs, lr, device):
    for p in model.input.parameters(): p.requires_grad = False
    for p in model.output.parameters(): p.requires_grad = False
    for p in model.field.parameters(): p.requires_grad = True
    opt = torch.optim.Adam(model.field.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    for epoch in range(epochs):
        model.train(); correct = total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            z = model(x); loss = F.cross_entropy(z, y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.field.parameters(), 1.0); opt.step()
            correct += (z.argmax(1) == y).sum().item(); total += y.numel()
        history.append({'epoch': epoch + 1, 'train_acc': correct / total, 'test_acc': accuracy(model, test_loader, device)})
    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pretrain-epochs', type=int, default=20)
    p.add_argument('--field-epochs', type=int, default=20)
    p.add_argument('--subset', type=int, default=5000)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cpu')
    p.add_argument('--result-tag', default='seed0_20plus20')
    a = p.parse_args(); torch.set_num_threads(1); torch.manual_seed(a.seed)
    device = torch.device(a.device)
    train_loader, test_loader = load_data('data/mnist', a.batch, a.subset)

    base = LocalElectricalNFV3MLP(8, 8, 4, mode='raw_bounded').to(device)
    opt = torch.optim.Adam(base.parameters(), lr=a.lr, weight_decay=1e-4)
    best_acc = -1.0; best_state = None; pretrain = []
    for epoch in range(a.pretrain_epochs):
        base.train(); correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device); z = base(x); loss = F.cross_entropy(z, y)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0); opt.step()
            correct += (z.argmax(1) == y).sum().item(); total += y.numel()
        test_acc = accuracy(base, test_loader, device)
        pretrain.append({'epoch': epoch + 1, 'train_acc': correct / total, 'test_acc': test_acc})
        if test_acc > best_acc: best_acc, best_state = test_acc, copy.deepcopy(base.state_dict())
    base.load_state_dict(best_state)

    def make_field(kind):
        common = dict(height=8, width=8, steps=4, mode='raw_bounded', threshold_init=.5, strength_init=.5, decay_init=.8, tau=.2)
        if kind == 'baseline': return LocalElectricalFieldV3(**common, collect_diagnostics=False)
        if kind == 'learnable_kernel': return LearnableKernelField(**common)
        if kind == 'local_interaction': return InteractionField(**common)
        if kind == 'local_gate': return GatedField(**common)
        raise ValueError(kind)

    results = {}
    start = time.perf_counter()
    for kind in ('baseline', 'learnable_kernel', 'local_interaction', 'local_gate'):
        field = make_field(kind).to(device)
        field.load_state_dict(base.field.state_dict(), strict=False)
        model = FieldOnlyModel(base.input, base.output, field).to(device)
        initial = {n: p.detach().clone() for n, p in field.named_parameters()}
        history = train_one(model, train_loader, test_loader, a.field_epochs, a.lr, device)
        delta = {n: float((p.detach() - initial[n]).abs().mean()) for n, p in field.named_parameters()}
        results[kind] = {
            'best_test_acc': max(r['test_acc'] for r in history),
            'final_test_acc': history[-1]['test_acc'],
            'parameters_total': sum(p.numel() for p in model.parameters()),
            'parameters_field': sum(p.numel() for p in field.parameters()),
            'history': history,
            'field_abs_delta_mean': delta,
        }
        print(f"{kind:18s} best={results[kind]['best_test_acc']:.4f} final={results[kind]['final_test_acc']:.4f} field_params={results[kind]['parameters_field']}")
    result = {'config': vars(a), 'pretrain_best_test_acc': best_acc, 'pretrain_history': pretrain, 'variants': results, 'seconds_total': time.perf_counter() - start}
    out = 'local_electrical_nf_usage_results'; os.makedirs(out, exist_ok=True)
    path = os.path.join(out, 'field_expressive_variants_' + a.result_tag + '.json')
    with open(path, 'w', encoding='utf-8') as f: json.dump(result, f, indent=2)
    print('saved', path)


if __name__ == '__main__': main()
