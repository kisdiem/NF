"""Local Electrical NF v2 with activity-dependent local inhibition.

This is intentionally separate from local_electrical_nf.py.  It keeps the
fixed 8x8 field and local conv2d propagation, but removes permanent
excitatory/inhibitory node signs.  All releases are excitatory; inhibition is
generated from the local release activity.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalElectricalFieldV2(nn.Module):
    def __init__(self, height=8, width=8, steps=4, threshold_init=0.5,
                 strength_init=0.5, decay_init=0.8, tau=0.2,
                 dynamic_inhibition=True, refractory=False,
                 rho_init=0.15, beta_init=1.0, tau_inhibition=0.2,
                 lambda_r=0.8, gamma_init=0.5):
        super().__init__()
        self.height, self.width, self.steps = height, width, steps
        self.tau = tau
        self.dynamic_inhibition = dynamic_inhibition
        self.refractory = refractory
        self.tau_inhibition = tau_inhibition
        self.lambda_r = lambda_r
        self.theta_raw = nn.Parameter(torch.full(
            (1, 1, height, width), math.log(math.expm1(threshold_init))))
        self.strength_raw = nn.Parameter(torch.full(
            (1, 1, height, width), math.log(math.expm1(strength_init))))
        self.decay_raw = nn.Parameter(torch.full(
            (1, 1, height, width), math.log(decay_init / (1 - decay_init))))
        # Shared positive inhibition parameters, deliberately not per-node.
        self.rho_raw = nn.Parameter(torch.tensor(math.log(math.expm1(rho_init))))
        self.beta_raw = nn.Parameter(torch.tensor(math.log(math.expm1(beta_init))))
        self.gamma_raw = nn.Parameter(torch.tensor(math.log(math.expm1(gamma_init))))
        k_exc = torch.tensor([[0.7, 1.0, 0.7],
                              [1.0, 0.0, 1.0],
                              [0.7, 1.0, 0.7]], dtype=torch.float32)
        k_inh = torch.ones(3, 3, dtype=torch.float32)
        k_inh[1, 1] = 0.0
        k_inh /= k_inh.sum()
        self.register_buffer("exc_kernel", k_exc.view(1, 1, 3, 3))
        self.register_buffer("inh_kernel", k_inh.view(1, 1, 3, 3))
        self.last_diagnostics = {}
        self.last_states, self.last_releases = [], []
        self.last_excitations, self.last_inhibitions = [], []

    def _effective_parameters(self):
        theta = F.softplus(self.theta_raw)
        strength = F.softplus(self.strength_raw)
        decay = torch.sigmoid(self.decay_raw)
        rho = F.softplus(self.rho_raw)
        beta = F.softplus(self.beta_raw)
        gamma = F.softplus(self.gamma_raw)
        return theta, strength, decay, rho, beta, gamma

    def forward(self, x):
        expected = (1, self.height, self.width)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"expected (B,1,{self.height},{self.width}), got {tuple(x.shape)}")
        theta, strength, decay, rho, beta, gamma = self._effective_parameters()
        v = x
        r_state = torch.zeros_like(v)
        states, releases, excitations, inhibitions = [], [], [], []
        changes, gates, refractory_values, effective_thresholds = [], [], [], []
        for _ in range(self.steps):
            if self.refractory:
                theta_eff = theta + gamma * r_state
            else:
                theta_eff = theta
            gate = torch.sigmoid((v - theta_eff) / self.tau)
            release = gate * strength
            excitation = F.conv2d(release, self.exc_kernel, padding=1)
            local_activity = F.conv2d(release, self.inh_kernel, padding=1)
            if self.dynamic_inhibition:
                inh_gate = torch.sigmoid((local_activity - rho) / self.tau_inhibition)
                inhibition = beta * inh_gate * local_activity
            else:
                inhibition = torch.zeros_like(excitation)
            incoming = excitation - inhibition
            v_next = decay * v + incoming
            changes.append((v_next - v).abs().mean())
            states.append(v_next); releases.append(release)
            excitations.append(excitation); inhibitions.append(inhibition)
            gates.append(gate)
            if self.refractory:
                r_next = self.lambda_r * r_state + gate
                refractory_values.append(r_next)
            else:
                r_next = r_state
                refractory_values.append(r_state)
            effective_thresholds.append(theta_eff)
            v, r_state = v_next, r_next

        self.last_states = [s.detach()[:1] for s in states]
        self.last_releases = [s.detach()[:1] for s in releases]
        self.last_excitations = [s.detach()[:1] for s in excitations]
        self.last_inhibitions = [s.detach()[:1] for s in inhibitions]
        self.last_diagnostics = {
            "membrane_mean": [s.mean().item() for s in states],
            "membrane_abs_mean": [s.abs().mean().item() for s in states],
            "state_change": torch.stack(changes).detach().cpu().tolist(),
            "activation_rate": [g.gt(0.5).float().mean().item() for g in gates],
            "release_abs_mean": [r.abs().mean().item() for r in releases],
            "excitation_mean": [e.mean().item() for e in excitations],
            "inhibition_mean": [i.mean().item() for i in inhibitions],
            "inhibition_gate_mean": [
                (i / (beta * a + 1e-8)).mean().item()
                for i, a in zip(inhibitions, [F.conv2d(r, self.inh_kernel, padding=1) for r in releases])
            ],
            "threshold_mean": theta.mean().item(),
            "effective_threshold_mean": [t.mean().item() for t in effective_thresholds],
            "decay_mean": decay.mean().item(),
            "refractory_mean": [r.mean().item() for r in refractory_values],
            "rho": rho.item(), "beta": beta.item(), "gamma": gamma.item(),
            "dead_ratio": (v.abs() < 1e-4).float().mean().item(),
            "saturated_ratio": (v.abs() > 10).float().mean().item(),
        }
        return v

    def parameter_report(self):
        return {
            "parameters": sum(p.numel() for p in self.parameters()),
            "field_parameters": sum(p.numel() for p in self.parameters()),
            "local_kernel_size": 3,
            "local_neighbor_positions": 8,
            "approx_local_propagation_flops_per_step": self.height * self.width * 9 * 2 * (2 if self.dynamic_inhibition else 1),
            "steps": self.steps,
            "has_nxn_relation": False,
            "complexity": "O(N*k*T), k=8 local neighbors",
        }


class LocalElectricalNFV2MLP(nn.Module):
    def __init__(self, height=8, width=8, steps=4, **field_cfg):
        super().__init__()
        if height * width != 64:
            raise ValueError("first version expects height*width=64")
        self.input = nn.Linear(784, 64)
        self.field = LocalElectricalFieldV2(height, width, steps, **field_cfg)
        self.output = nn.Linear(64, 10)

    def forward(self, x, return_hidden=False):
        h = self.input(x.flatten(1)).view(-1, 1, 8, 8)
        h = self.field(h)
        flat = h.flatten(1)
        y = self.output(flat)
        return (y, flat) if return_hidden else y
