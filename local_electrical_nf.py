"""Local Electrical Neural Field.

Unlike DynamicNeuralField, this field has fixed 2-D node positions and uses a
fixed local 3x3 electrical kernel.  Propagation is implemented by conv2d, so
there is no Q/K, attention, N*N relation matrix, or per-node Python loop.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalElectricalField(nn.Module):
    def __init__(self, height=8, width=8, steps=4, threshold_init=0.5,
                 strength_init=0.5, decay_init=0.8, tau=0.2,
                 no_threshold=False, persistence=True, inhibition=True):
        super().__init__()
        self.height, self.width, self.steps = height, width, steps
        self.tau = tau
        self.no_threshold = no_threshold
        self.persistence = persistence
        self.inhibition = inhibition
        self.theta_raw = nn.Parameter(torch.full(
            (1, 1, height, width), math.log(math.expm1(threshold_init))))
        self.strength_raw = nn.Parameter(torch.full(
            (1, 1, height, width), math.atanh(strength_init)))
        self.decay_raw = nn.Parameter(torch.full(
            (1, 1, height, width), math.log(decay_init / (1 - decay_init))))
        # Signed strength represents excitation (+) or inhibition (-).
        self.sign_raw = nn.Parameter(torch.zeros(1, 1, height, width))
        with torch.no_grad():
            signs = torch.ones(1, 1, height, width)
            signs[..., 1::2] = -1.0
            self.sign_raw.copy_(2.0 * signs)
        k = torch.tensor([[0.7, 1.0, 0.7],
                          [1.0, 0.0, 1.0],
                          [0.7, 1.0, 0.7]], dtype=torch.float32)
        self.register_buffer("local_kernel", k.view(1, 1, 3, 3))
        self.last_diagnostics = {}
        self.last_states = []
        self.last_releases = []

    def _effective_parameters(self):
        theta = F.softplus(self.theta_raw)
        strength = torch.tanh(self.strength_raw)
        # sign_raw learns whether a node is excitatory or inhibitory; tanh
        # keeps the multiplier bounded and permits both signs.
        signed_strength = strength * torch.tanh(self.sign_raw + 1e-6)
        if not self.inhibition:
            signed_strength = signed_strength.abs()
        decay = torch.sigmoid(self.decay_raw)
        return theta, signed_strength, decay

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (1, self.height, self.width):
            raise ValueError(f"expected (B,1,{self.height},{self.width}), got {tuple(x.shape)}")
        theta, signed_strength, decay = self._effective_parameters()
        v = x
        states, releases = [], []
        changes = []
        for _ in range(self.steps):
            if self.no_threshold:
                release_gate = torch.ones_like(v)
            else:
                release_gate = torch.sigmoid((v - theta) / self.tau)
            release = release_gate * signed_strength
            incoming = F.conv2d(release, self.local_kernel, padding=1)
            if self.persistence:
                v_next = decay * v + incoming
            else:
                v_next = incoming
            changes.append((v_next - v).abs().mean())
            states.append(v_next)
            releases.append(release)
            v = v_next
        self.last_states = [s.detach()[:1] for s in states]
        self.last_releases = [s.detach()[:1] for s in releases]
        self.last_diagnostics = {
            "membrane_abs_mean": [s.abs().mean().item() for s in states],
            "membrane_mean": [s.mean().item() for s in states],
            "state_change": torch.stack(changes).detach().cpu().tolist(),
            "activation_rate": [r.abs().gt(1e-4).float().mean().item() for r in releases],
            "release_abs_mean": [r.abs().mean().item() for r in releases],
            "threshold_mean": theta.mean().item(),
            "strength_abs_mean": signed_strength.abs().mean().item(),
            "positive_strength_ratio": (signed_strength > 0).float().mean().item(),
            "negative_strength_ratio": (signed_strength < 0).float().mean().item(),
            "decay_mean": decay.mean().item(),
            "dead_ratio": (v.abs() < 1e-4).float().mean().item(),
            "saturated_ratio": (v.abs() > 10).float().mean().item(),
        }
        return v

    def parameter_report(self):
        return {
            "parameters": sum(p.numel() for p in self.parameters()),
            "local_kernel_size": 3,
            "local_neighbor_positions": 8,
            "approx_flops_per_step": self.height * self.width * 9 * 2,
            "steps": self.steps,
            "has_nxn_relation": False,
        }


class LocalElectricalNFMLP(nn.Module):
    def __init__(self, height=8, width=8, steps=4, **field_cfg):
        super().__init__()
        if height * width != 64:
            raise ValueError("first version expects height*width=64")
        self.input = nn.Linear(784, 64)
        self.field = LocalElectricalField(height, width, steps, **field_cfg)
        self.output = nn.Linear(64, 10)

    def forward(self, x, return_hidden=False):
        h = self.input(x.flatten(1)).view(-1, 1, 8, 8)
        h = self.field(h)
        flat = h.flatten(1)
        y = self.output(flat)
        return (y, flat) if return_hidden else y
