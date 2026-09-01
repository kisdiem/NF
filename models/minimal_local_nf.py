"""Minimal local neural field used by intrinsic-training experiments.

This preserves the current A design: membrane state, one shared decay, and a
shared local connection kernel.  The field records detached dynamics for
diagnostics but does not add losses or change the forward mapping.
"""
import torch
from torch import nn
import torch.nn.functional as F


class MinimalLocalField(nn.Module):
    def __init__(self, size=16, steps=1, circular=False):
        super().__init__()
        self.size = int(size)
        self.steps = int(steps)
        self.circular = bool(circular)
        self.decay_raw = nn.Parameter(torch.tensor(1.38629436112))
        self.kernel = nn.Parameter(
            torch.tensor([[0.0, 0.2, 0.0],
                          [0.2, 0.0, 0.2],
                          [0.0, 0.2, 0.0]]).view(1, 1, 3, 3))
        self.last_states = []
        self.last_activations = []
        self.last_diagnostics = {}

    @property
    def decay(self):
        return torch.sigmoid(self.decay_raw)

    def forward(self, x):
        v = x
        states, activations, changes = [], [], []
        for _ in range(self.steps):
            signal = torch.tanh(v)
            if self.circular:
                incoming = F.conv2d(F.pad(signal, (1, 1, 1, 1), mode="circular"),
                                    self.kernel)
            else:
                incoming = F.conv2d(signal, self.kernel, padding=1)
            v_next = self.decay * v + incoming
            changes.append((v_next - v).abs().mean())
            states.append(v_next)
            activations.append(signal)
            v = v_next
        self.last_states = [s.detach()[:1] for s in states]
        self.last_activations = [a.detach()[:1] for a in activations]
        self.last_diagnostics = {
            "state_abs_mean": [s.detach().abs().mean().item() for s in states],
            "state_variance": [s.detach().var(unbiased=False).item() for s in states],
            "state_change": ([x.detach().item() for x in changes] if changes else []),
            "activation_mean": ([a.detach().abs().mean().item() for a in activations]
                                if activations else []),
            "dead_ratio": (float((v.detach().abs() < 1e-4).float().mean())
                           if self.steps else 0.0),
            "saturated_ratio": (float((v.detach().abs() > 10).float().mean())
                                if self.steps else 0.0),
            "decay": float(self.decay.detach()),
        }
        return v


class MinimalLocalNFMLP(nn.Module):
    def __init__(self, hidden=256, steps=1, circular=False):
        super().__init__()
        size = int(hidden ** 0.5)
        if size * size != hidden:
            raise ValueError("hidden must be a square number")
        self.input = nn.Linear(784, hidden)
        self.field = MinimalLocalField(size=size, steps=steps, circular=circular)
        self.output = nn.Linear(hidden, 10)

    def forward(self, x):
        h = self.input(x.flatten(1)).view(-1, 1, self.field.size, self.field.size)
        return self.output(self.field(h).flatten(1))
