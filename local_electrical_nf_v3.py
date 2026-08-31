"""Stability-oriented Local Electrical NF experiments.

Independent from v1/v2.  The main change is matching the scale of excitation
and inhibition: the excitatory stencil can be normalized to unit mass.
Optional bounded input and homeostatic leak mechanisms are explicit variants.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalElectricalFieldV3(nn.Module):
    def __init__(self, height=8, width=8, steps=4, mode="normalized",
                 threshold_init=0.5, strength_init=0.5, decay_init=0.8,
                 tau=0.2, rho_init=0.15, beta_init=1.0,
                 tau_inhibition=0.2, lambda_r=0.8, gamma_init=0.5,
                 excitation_gain=1.0, fuse_local_convs=False,
                 collect_diagnostics=True):
        super().__init__()
        self.height, self.width, self.steps, self.mode = height, width, steps, mode
        self.tau, self.tau_inhibition, self.lambda_r = tau, tau_inhibition, lambda_r
        self.excitation_gain = excitation_gain
        self.fuse_local_convs = fuse_local_convs
        self.collect_diagnostics = collect_diagnostics
        self.theta_raw = nn.Parameter(torch.full((1, 1, height, width), math.log(math.expm1(threshold_init))))
        self.strength_raw = nn.Parameter(torch.full((1, 1, height, width), math.log(math.expm1(strength_init))))
        self.decay_raw = nn.Parameter(torch.full((1, 1, height, width), math.log(decay_init / (1 - decay_init))))
        self.rho_raw = nn.Parameter(torch.tensor(math.log(math.expm1(rho_init))))
        self.beta_raw = nn.Parameter(torch.tensor(math.log(math.expm1(beta_init))))
        self.gamma_raw = nn.Parameter(torch.tensor(math.log(math.expm1(gamma_init))))
        k_exc = torch.tensor([[0.7, 1.0, 0.7], [1.0, 0.0, 1.0], [0.7, 1.0, 0.7]])
        k_inh = torch.ones(3, 3); k_inh[1, 1] = 0; k_inh /= k_inh.sum()
        if mode not in ("raw", "raw_bounded", "raw_unbounded"): k_exc /= k_exc.sum()
        self.register_buffer("exc_kernel", k_exc.float().view(1, 1, 3, 3))
        self.register_buffer("inh_kernel", k_inh.float().view(1, 1, 3, 3))
        self.register_buffer("fused_kernel", torch.stack((k_exc, k_inh)).float().view(2, 1, 3, 3))
        self.last_diagnostics = {}; self.last_states = []; self.last_releases = []
        self.last_excitations = []; self.last_inhibitions = []

    def _effective_parameters(self):
        return (F.softplus(self.theta_raw), F.softplus(self.strength_raw),
                torch.sigmoid(self.decay_raw), F.softplus(self.rho_raw),
                F.softplus(self.beta_raw), F.softplus(self.gamma_raw))

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, self.height, self.width):
            raise ValueError(f"expected (B,1,{self.height},{self.width}), got {tuple(x.shape)}")
        theta, strength, decay, rho, beta, gamma = self._effective_parameters()
        v = x; r_state = torch.zeros_like(v)
        states=[]; releases=[]; excitations=[]; inhibitions=[]; activities=[]; changes=[]; gates=[]; rs=[]; ets=[]
        for _ in range(self.steps):
            theta_eff = theta + gamma * r_state if self.mode in ("refractory", "refractory_bounded", "centered_refractory") else theta
            gate = torch.sigmoid((v - theta_eff) / self.tau)
            release = gate * strength
            if self.fuse_local_convs:
                local_maps = F.conv2d(release, self.fused_kernel, padding=1)
                excitation = self.excitation_gain * local_maps[:, 0:1]
                activity = local_maps[:, 1:2]
            else:
                excitation = self.excitation_gain * F.conv2d(release, self.exc_kernel, padding=1)
                activity = F.conv2d(release, self.inh_kernel, padding=1)
            inh_gate = torch.sigmoid((activity - rho) / self.tau_inhibition)
            inhibition = torch.zeros_like(excitation) if self.mode == "raw_unbounded" else beta * inh_gate * activity
            incoming = excitation - inhibition
            if self.mode in ("bounded", "refractory_bounded", "raw_bounded"):
                incoming = torch.tanh(incoming)
            if self.mode in ("centered", "centered_refractory"):
                incoming = incoming - incoming.mean(dim=(2, 3), keepdim=True)
            if self.mode == "homeostatic":
                # Local excess activity produces an additional smooth leak.
                incoming = incoming - 0.5 * F.relu(activity - rho)
            v_next = decay * v + incoming
            if self.mode == "soft_reset":
                v_next = v_next - 0.5 * gate
            if self.collect_diagnostics:
                changes.append((v_next-v).abs().mean())
                states.append(v_next); releases.append(release); excitations.append(excitation); inhibitions.append(inhibition); activities.append(activity)
                gates.append(gate); ets.append(theta_eff)
            r_next = self.lambda_r * r_state + gate if self.mode in ("refractory", "refractory_bounded", "centered_refractory") else r_state
            if self.collect_diagnostics: rs.append(r_next)
            v, r_state = v_next, r_next
        if not self.collect_diagnostics:
            self.last_diagnostics = {}
            self.last_states = []; self.last_releases = []
            self.last_excitations = []; self.last_inhibitions = []
            return v
        self.last_states=[s.detach()[:1] for s in states]; self.last_releases=[s.detach()[:1] for s in releases]
        self.last_excitations=[s.detach()[:1] for s in excitations]; self.last_inhibitions=[s.detach()[:1] for s in inhibitions]
        self.last_diagnostics={
            "membrane_mean":[s.mean().item() for s in states], "membrane_abs_mean":[s.abs().mean().item() for s in states],
            "state_change":torch.stack(changes).detach().cpu().tolist(),
            "activation_rate":[g.gt(0.5).float().mean().item() for g in gates],
            "release_abs_mean":[r.abs().mean().item() for r in releases],
            "excitation_mean":[e.mean().item() for e in excitations], "inhibition_mean":[i.mean().item() for i in inhibitions],
            "inhibition_gate_mean":[(i/(beta*a+1e-8)).mean().item() for i,a in zip(inhibitions,activities)],
            "threshold_mean":theta.mean().item(), "effective_threshold_mean":[t.mean().item() for t in ets],
            "decay_mean":decay.mean().item(), "refractory_mean":[r.mean().item() for r in rs],
            "rho":rho.item(), "beta":beta.item(), "gamma":gamma.item(),
            "dead_ratio":(v.abs()<1e-4).float().mean().item(), "saturated_ratio":(v.abs()>10).float().mean().item(),
        }
        return v

    def parameter_report(self):
        return {"parameters":sum(p.numel() for p in self.parameters()), "field_parameters":sum(p.numel() for p in self.parameters()),
                "local_neighbor_positions":8, "excitation_normalized":self.mode != "raw",
                "excitation_gain":self.excitation_gain,
                "fused_local_convs":self.fuse_local_convs,
                "approx_local_propagation_flops_per_step":self.height*self.width*9*2*2,
                "has_nxn_relation":False, "complexity":"O(N*k*T), k=8"}


class LocalElectricalNFV3MLP(nn.Module):
    def __init__(self, height=8, width=8, steps=4, mode="normalized", **field_cfg):
        super().__init__(); self.input=nn.Linear(784,64); self.field=LocalElectricalFieldV3(height,width,steps,mode=mode,**field_cfg); self.output=nn.Linear(64,10)
    def forward(self,x):
        h=self.input(x.flatten(1)).view(-1,1,8,8); return self.output(self.field(h).flatten(1))
