"""Temporal benchmarks for the current Dynamic NF.

The field is reused as a recurrent cell: each external time step injects a
new input into a persistent [B,16,4] state, then performs one synchronous
Dynamic NF update.  No Transformer/GNN implementation is used.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamic_nf import DynamicNeuralField


class TemporalModel(nn.Module):
    def __init__(self, task="seq", kind="dynamic", hidden=64, pixelwise=False,
                 permutation_seed=0):
        super().__init__()
        self.task, self.kind, self.hidden = task, kind, hidden
        self.pixelwise = pixelwise
        if task == "moving":
            self.step_dim = 36 * 36
        else:
            self.step_dim = 1 if pixelwise else 28
        self.input = nn.Linear(self.step_dim, hidden)
        self.gru = nn.GRU(self.step_dim, hidden, batch_first=True) if kind == "gru" else None
        if kind == "dynamic":
            self.field = DynamicNeuralField(16, 4, branches=4, steps=1,
                                            relation_gain_init=0.1,
                                            temperature=1.0, norm=True)
        self.output = nn.Linear(hidden, 10)
        if task == "perm":
            g = torch.Generator().manual_seed(permutation_seed)
            self.register_buffer("permutation", torch.randperm(784, generator=g))

    @staticmethod
    def shift_frames(x):
        # Ten deterministic moving frames from each 28x28 digit.
        b = x.shape[0]
        frames = []
        for t in range(10):
            dx, dy = t % 5, (t * 2) % 5
            canvas = torch.zeros(b, 1, 36, 36, device=x.device, dtype=x.dtype)
            canvas[:, :, dy:dy + 28, dx:dx + 28] = x
            frames.append(canvas.flatten(1))
        return torch.stack(frames, dim=1)

    def make_sequence(self, x):
        if self.task == "moving":
            return self.shift_frames(x)
        flat = x.flatten(1)
        if self.task == "perm":
            flat = flat[:, self.permutation]
        if self.pixelwise:
            return flat.unsqueeze(-1)
        return flat.view(-1, 28, 28)

    def forward(self, x, return_hidden=False):
        seq = self.make_sequence(x)
        if self.kind == "gru":
            _, h = self.gru(seq)
            hidden = h[-1]
        else:
            state = torch.zeros(x.shape[0], 16, 4, device=x.device, dtype=x.dtype)
            changes, relation_abs = [], []
            for t in range(seq.shape[1]):
                injected = self.input(seq[:, t]).view(-1, 16, 4)
                old = state
                state = 0.8 * state + 0.2 * injected
                if self.kind == "linear":
                    pass
                elif self.kind == "relu":
                    state = F.relu(state)
                else:
                    state = self.field(state)
                    relation_abs.append(self.field.last_diagnostics.get(
                        "final_relation_abs_mean", 0.0))
                changes.append((state - old).abs().mean().detach())
            hidden = state.flatten(1)
            if self.kind == "dynamic":
                self.temporal_diagnostics = {
                    "state_change": torch.stack(changes).cpu().tolist(),
                    "relation_abs_mean": relation_abs,
                    "last_relations": self.field.last_relations,
                }
        out = self.output(hidden)
        return (out, hidden) if return_hidden else out


def make_temporal_dataset(root, task, subset=5000, seed=0):
    from torchvision import datasets, transforms
    ds = datasets.MNIST(root=root, train=True, download=True, transform=transforms.ToTensor())
    if subset:
        ids = torch.arange(min(subset, len(ds)))
        ds = torch.utils.data.Subset(ds, ids)
    return ds
