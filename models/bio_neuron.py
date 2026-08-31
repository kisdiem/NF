"""Minimal differentiable bio-neuron model.

The implementation is intentionally small: branch-local computation,
excitatory/inhibitory weights, short temporal traces, soma integration and an
adaptive threshold.  All neuron/branch computation is batched with einsum;
the only loop is over the requested small number of time steps.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BioNeuronLayer(nn.Module):
    def __init__(self, d_in, n_neurons, branches=4, steps=3,
                 dendrite="soft_threshold", temporal=True,
                 inhibition=True, adaptive_threshold=True,
                 trace_decay=0.8, dendrite_decay=0.8, membrane_decay=0.8,
                 adaptation_decay=0.9, threshold_init=0.5,
                 adaptation_init=0.1, gate_sharpness=4.0,
                 output_mode="mean", hard_spike=False, weight_rank=0):
        super().__init__()
        if dendrite not in ("soft_threshold", "quadratic", "tanh"):
            raise ValueError(dendrite)
        if output_mode not in ("mean", "final", "membrane"):
            raise ValueError(output_mode)
        self.d_in, self.n_neurons, self.branches = d_in, n_neurons, branches
        self.steps = steps
        self.dendrite = dendrite
        self.temporal = temporal
        self.inhibition = inhibition
        self.adaptive_threshold = adaptive_threshold
        self.trace_decay = trace_decay
        self.dendrite_decay = dendrite_decay
        self.membrane_decay = membrane_decay
        self.adaptation_decay = adaptation_decay
        self.gate_sharpness = gate_sharpness
        self.output_mode = output_mode
        self.hard_spike = hard_spike
        self.weight_rank = int(weight_rank)

        shape = (n_neurons, branches, d_in)
        # Positive parameterizations make excitation/inhibition interpretable.
        # Low-rank mode replaces N*B*Din weights with branch bases and
        # neuron-specific coefficients.
        if self.weight_rank > 0:
            r = self.weight_rank
            self.exc_left_raw = nn.Parameter(torch.empty(n_neurons, branches, r))
            self.exc_right_raw = nn.Parameter(torch.empty(branches, r, d_in))
            self.inh_left_raw = nn.Parameter(torch.empty(n_neurons, branches, r))
            self.inh_right_raw = nn.Parameter(torch.empty(branches, r, d_in))
            for p in (self.exc_left_raw, self.exc_right_raw,
                      self.inh_left_raw, self.inh_right_raw):
                nn.init.normal_(p, mean=-1.2, std=0.25)
        else:
            self.exc_raw = nn.Parameter(torch.empty(shape))
            self.inh_raw = nn.Parameter(torch.empty(shape))
            nn.init.normal_(self.exc_raw, mean=-1.2, std=0.25)
            nn.init.normal_(self.inh_raw, mean=-1.2, std=0.25)
        self.branch_bias = nn.Parameter(torch.zeros(n_neurons, branches))
        self.branch_gain_raw = nn.Parameter(torch.zeros(n_neurons, branches))
        self.soma_gain_raw = nn.Parameter(torch.zeros(n_neurons))
        self.theta_raw = nn.Parameter(torch.full(
            (n_neurons,), math.log(math.expm1(threshold_init))))
        self.adaptation_raw = nn.Parameter(torch.full(
            (n_neurons,), math.log(math.expm1(adaptation_init))))
        # Start with balanced excitation and inhibition.  The previous
        # initialization made almost every effective weight positive, which
        # pushed the soma into one-sided saturation on MNIST.
        self.last_diagnostics = {}

    def effective_weights(self):
        if self.weight_rank > 0:
            exc = torch.einsum("nbr,brd->nbd",
                               F.softplus(self.exc_left_raw),
                               F.softplus(self.exc_right_raw))
            inh = torch.einsum("nbr,brd->nbd",
                               F.softplus(self.inh_left_raw),
                               F.softplus(self.inh_right_raw))
            return exc, inh
        return F.softplus(self.exc_raw), F.softplus(self.inh_raw)

    def _activation(self, z):
        if self.dendrite == "soft_threshold":
            # Signed soft threshold: both excitation and inhibition can reach
            # the soma, while small inputs are suppressed.
            return torch.tanh(self.gate_sharpness * z) * torch.sigmoid(
                self.gate_sharpness * (z.abs() - 0.25))
        if self.dendrite == "quadratic":
            z = z.clamp(-4.0, 4.0)
            return z.sign() * z.abs().square() / 2.0
        return torch.tanh(self.gate_sharpness * z)

    def _spike(self, v, theta):
        soft = torch.sigmoid(self.gate_sharpness * (v - theta))
        if not self.hard_spike:
            return soft
        hard = (v >= theta).to(v.dtype)
        return hard + soft - soft.detach()

    def forward(self, x):
        if x.ndim != 2 or x.shape[-1] != self.d_in:
            raise ValueError(f"expected (B,{self.d_in}), got {tuple(x.shape)}")
        bsz = x.shape[0]
        exc, inh_full = self.effective_weights()
        inh = inh_full if self.inhibition else torch.zeros_like(exc)
        signed_w = exc - inh
        branch_gain = F.softplus(self.branch_gain_raw) + 1e-3
        soma_gain = F.softplus(self.soma_gain_raw) + 1e-3
        theta_base = F.softplus(self.theta_raw)
        adaptation_strength = (F.softplus(self.adaptation_raw)
                               if self.adaptive_threshold
                               else torch.zeros_like(theta_base))

        trace = torch.zeros(bsz, self.d_in, device=x.device, dtype=x.dtype)
        d_state = torch.zeros(bsz, self.n_neurons, self.branches,
                              device=x.device, dtype=x.dtype)
        v = torch.zeros(bsz, self.n_neurons, device=x.device, dtype=x.dtype)
        a_state = torch.zeros_like(v)
        prev_s = torch.zeros_like(v)
        outputs = []
        d_values = []
        v_values = []
        theta_values = []
        a_values = []

        for _ in range(self.steps):
            current = (self.trace_decay * trace + x
                       if self.temporal else x)
            trace = current
            z = torch.einsum("bd,nrd->bnr", current, signed_w)
            z = z + self.branch_bias.unsqueeze(0)
            d_now = self._activation(z) * branch_gain.unsqueeze(0)
            d_state = (self.dendrite_decay * d_state + d_now
                       if self.temporal else d_now)
            # A soma averages branch currents.  Summing four branches and
            # then accumulating over time made the first MNIST version enter
            # a large negative potential before gradients could act.
            soma = (d_state * soma_gain.unsqueeze(-1)).mean(dim=-1)
            v = self.membrane_decay * v + soma
            a_state = self.adaptation_decay * a_state + prev_s
            theta = theta_base + adaptation_strength * a_state
            s = self._spike(v, theta)
            prev_s = s
            outputs.append(s)
            d_values.append(d_state)
            v_values.append(v)
            theta_values.append(theta)
            a_values.append(a_state)

        if self.output_mode == "final":
            out = outputs[-1]
        elif self.output_mode == "membrane":
            # Signed distance from the adaptive threshold; no input bypass.
            out = v - theta
        else:
            out = torch.stack(outputs).mean(0)
        d_all = torch.stack(d_values)
        v_all = torch.stack(v_values)
        theta_all = torch.stack(theta_values)
        a_all = torch.stack(a_values)
        self.last_diagnostics = {
            "dendritic_activation_mean": d_all.abs().mean().item(),
            "dendritic_activation_std": d_all.std().item(),
            "soma_v_mean": v_all.mean().item(),
            "soma_v_std": v_all.std().item(),
            "threshold_mean": theta_all.mean().item(),
            "threshold_std": theta_all.std().item(),
            "activation_rate": out.mean().item(),
            "adaptation_mean": a_all.mean().item(),
            "dead_neuron_ratio": (out.abs().mean(0) < 1e-4).float().mean().item(),
            "saturated_neuron_ratio": ((out.mean(0) < 0.01) | (out.mean(0) > 0.99)).float().mean().item(),
            "positive_weight_ratio": (signed_w > 0).float().mean().item(),
            "negative_weight_ratio": (signed_w < 0).float().mean().item(),
        }
        return out

    def parameter_report(self):
        return {
            "parameters": sum(p.numel() for p in self.parameters()),
            "approx_flops_per_step": self.n_neurons * self.branches * self.d_in * 2,
            "steps": self.steps,
        }


class BioMLP(nn.Module):
    """Input projection -> one BioNeuronLayer -> linear classifier."""
    def __init__(self, d_in, hidden, d_out, **bio_cfg):
        super().__init__()
        self.input = nn.Linear(d_in, hidden)
        self.bio = BioNeuronLayer(hidden, hidden, **bio_cfg)
        self.classifier = nn.Linear(hidden, d_out)

    def forward(self, x, return_hidden=False):
        x = x.flatten(1)
        hidden = self.bio(self.input(x))
        out = self.classifier(hidden)
        return (out, hidden) if return_hidden else out
