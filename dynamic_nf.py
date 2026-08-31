"""Dynamic Neural Field: state-dependent, synchronous relation dynamics.

This is an experiment built beside the original NF implementation.  It is not
an attention/Transformer module: relations are signed, unnormalised and are
recomputed from the evolving state at every step.  The field has one shared
node pool; there is no predeclared feed-forward tree.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicNeuralField(nn.Module):
    """H[B,N,D] -> H[B,N,D] with a dynamic directed relation matrix."""

    def __init__(self, n_nodes=16, node_dim=4, branches=4, steps=4,
                 relation_gain_init=0.1, temperature=1.0,
                 state_gate=True, state_persistence=True,
                 dynamic_relation=True, allow_feedback=True,
                 local_branches=True, norm=True, nonlinear=True,
                 strict_linear=False):
        super().__init__()
        self.n_nodes = n_nodes
        self.node_dim = node_dim
        self.branches = branches if local_branches else 1
        self.steps = steps
        self.state_gate = state_gate
        self.state_persistence = state_persistence
        self.dynamic_relation = dynamic_relation
        self.allow_feedback = allow_feedback
        self.local_branches = local_branches
        self.temperature = temperature
        self.norm_enabled = norm
        self.nonlinear = nonlinear
        self.strict_linear = strict_linear

        if not strict_linear:
            self.q_proj = nn.Linear(node_dim, node_dim, bias=False)
            self.k_proj = nn.Linear(node_dim, node_dim, bias=False)
            self.relation_gain_raw = nn.Parameter(
                torch.logit(torch.tensor(float(relation_gain_init))))
        else:
            self.static_relation = nn.Parameter(
                torch.randn(n_nodes, n_nodes) * relation_gain_init)
        self.self_proj = nn.Linear(node_dim, self.branches * node_dim)
        self.message_proj = nn.Linear(node_dim, self.branches * node_dim)
        self.branch_bias = nn.Parameter(torch.zeros(self.branches, node_dim))
        self.branch_mix = nn.Parameter(torch.full((self.branches,), 1.0 / self.branches))
        if state_gate:
            self.gate_proj = nn.Linear(2 * node_dim, node_dim)
        else:
            self.gate_proj = None
        self.update_proj = nn.Linear(2 * node_dim, node_dim)
        self.norm = nn.LayerNorm(node_dim) if norm else nn.Identity()
        self.reset_diagnostics()

    @property
    def relation_gain(self):
        # Bounded gain keeps recurrent feedback initially weak and stable.
        if self.strict_linear:
            return self.static_relation.abs().mean()
        if not self.nonlinear:
            return self.relation_gain_raw.abs()
        return torch.sigmoid(self.relation_gain_raw)

    def reset_diagnostics(self):
        self.last_diagnostics = {}
        self.last_relations = []
        self.last_states = []

    def _relation(self, h):
        if self.strict_linear:
            return self.static_relation.unsqueeze(0).expand(h.shape[0], -1, -1)
        q = self.q_proj(h)
        k = self.k_proj(h)
        score = torch.matmul(q, k.transpose(1, 2)) / (self.node_dim ** 0.5)
        relation = ((torch.tanh(score / max(self.temperature, 1e-6))
                     if self.nonlinear else score) * self.relation_gain)
        if not self.allow_feedback:
            # Source i may affect target j only in one fixed triangular half.
            # This is an ablation for feedback/cycles, not the default model.
            relation = torch.triu(relation, diagonal=1)
        return relation

    def forward(self, h):
        if h.ndim != 3 or h.shape[1:] != (self.n_nodes, self.node_dim):
            raise ValueError(
                f"expected (B,{self.n_nodes},{self.node_dim}), got {tuple(h.shape)}")
        self.reset_diagnostics()
        states = []
        relations = []
        changes = []
        relation_changes = []
        h0 = h
        fixed_relation = self._relation(h) if not self.dynamic_relation else None
        prev_relation = None
        for _ in range(self.steps):
            relation = self._relation(h) if self.dynamic_relation else fixed_relation
            # relation[i,j] means source i -> target j; update is synchronous.
            message = torch.einsum("bij,bid->bjd", relation, h)
            pair = torch.cat((h, message), dim=-1)
            if self.local_branches:
                self_part = self.self_proj(h).view(h.shape[0], self.n_nodes,
                                                   self.branches, self.node_dim)
                msg_part = self.message_proj(message).view(
                    h.shape[0], self.n_nodes, self.branches, self.node_dim)
                branch_raw = self_part + msg_part + self.branch_bias
                branch = torch.tanh(branch_raw) if self.nonlinear else branch_raw
                mix = (torch.softmax(self.branch_mix, dim=0)
                       if self.nonlinear else
                       torch.full_like(self.branch_mix, 1.0 / self.branches))
                candidate = (branch * mix.view(1, 1, -1, 1)).sum(dim=2)
            else:
                candidate_raw = self.update_proj(pair)
                candidate = (torch.tanh(candidate_raw) if self.nonlinear
                             else candidate_raw)
            if self.state_persistence:
                if self.state_gate:
                    gate = (torch.sigmoid(self.gate_proj(pair)) if self.nonlinear
                            else torch.full_like(h, 0.5))
                else:
                    gate = torch.full_like(h, 0.5)
                h_next = (1.0 - gate) * h + gate * candidate
            else:
                h_next = candidate
            h_next = self.norm(h_next) if self.nonlinear else h_next
            changes.append((h_next - h).abs().mean())
            if prev_relation is None:
                relation_changes.append(torch.zeros((), device=h.device, dtype=h.dtype))
            else:
                relation_changes.append((relation - prev_relation).abs().mean())
            states.append(h_next)
            relations.append(relation)
            prev_relation = relation
            h = h_next

        self.last_states = [x.detach()[:1] for x in states]
        self.last_relations = [x.detach()[:1] for x in relations]
        change_values = torch.stack(changes)
        relation_change_values = torch.stack(relation_changes)
        final_relation = relations[-1]
        self.last_diagnostics = {
            "state_change": change_values.detach().cpu().tolist(),
            "state_abs_mean": [x.abs().mean().item() for x in states],
            "relation_abs_mean": [x.abs().mean().item() for x in relations],
            "relation_positive_ratio": [(x > 0).float().mean().item() for x in relations],
            "relation_negative_ratio": [(x < 0).float().mean().item() for x in relations],
            "relation_near_zero_ratio": [(x.abs() < 0.01).float().mean().item() for x in relations],
            "relation_change": relation_change_values.detach().cpu().tolist(),
            "relation_gain": self.relation_gain.item(),
            "gradient_norm": None,
            "final_state_abs_mean": h.abs().mean().item(),
            "final_relation_abs_mean": final_relation.abs().mean().item(),
        }
        return h

    def parameter_report(self):
        return {
            "parameters": sum(p.numel() for p in self.parameters()),
            "approx_flops_per_step": 2 * self.n_nodes * self.n_nodes * self.node_dim
            + 2 * self.n_nodes * self.branches * self.node_dim * self.node_dim,
            "steps": self.steps,
        }


class DynamicNFMLP(nn.Module):
    """Fair 784 -> 64 -> DynamicNF -> 64 -> 10 path."""

    def __init__(self, d_in=784, hidden=64, d_out=10, n_nodes=16,
                 node_dim=4, **field_cfg):
        super().__init__()
        if n_nodes * node_dim != hidden:
            raise ValueError("n_nodes * node_dim must equal hidden")
        self.input = nn.Linear(d_in, hidden)
        self.field = DynamicNeuralField(n_nodes, node_dim, **field_cfg)
        self.output = nn.Linear(hidden, d_out)

    def forward(self, x, return_hidden=False):
        h = self.input(x.flatten(1)).view(-1, self.field.n_nodes,
                                           self.field.node_dim)
        h = self.field(h)
        flat = h.flatten(1)
        y = self.output(flat)
        return (y, flat) if return_hidden else y
