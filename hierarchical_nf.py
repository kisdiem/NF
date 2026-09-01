"""Two-level Dynamic Neural Field experiment.

The field contains one 8-node lower pool and one 8-node upper pool.  At every
step all relations are computed from the old states and both pools update
synchronously.  No graph library or sequential node loop is used.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalDynamicNeuralField(nn.Module):
    def __init__(self, n_nodes=8, node_dim=4, branches=4, steps=4,
                 relation_gain_init=0.1, temperature=1.0,
                 layer2_internal=True, feedback=False, feedback_gain_init=0.02,
                 norm=True):
        super().__init__()
        self.n_nodes, self.node_dim = n_nodes, node_dim
        self.branches, self.steps = branches, steps
        self.temperature, self.layer2_internal = temperature, layer2_internal
        self.feedback = feedback

        # Q/K are projected once per layer and reused by the cross relation.
        self.q1 = nn.Linear(node_dim, node_dim, bias=False)
        self.k1 = nn.Linear(node_dim, node_dim, bias=False)
        self.q2 = nn.Linear(node_dim, node_dim, bias=False)
        self.k2 = nn.Linear(node_dim, node_dim, bias=False)
        inv = lambda x: torch.logit(torch.tensor(float(x)))
        self.gain1_raw = nn.Parameter(inv(relation_gain_init))
        self.gain12_raw = nn.Parameter(inv(relation_gain_init))
        self.gain2_raw = nn.Parameter(inv(relation_gain_init))
        if feedback:
            self.gain21_raw = nn.Parameter(inv(feedback_gain_init))

        self.self1 = nn.Linear(node_dim, branches * node_dim)
        self.msg1 = nn.Linear(node_dim, branches * node_dim)
        self.self2 = nn.Linear(node_dim, branches * node_dim)
        self.msg2 = nn.Linear(node_dim, branches * node_dim)
        self.bias1 = nn.Parameter(torch.zeros(branches, node_dim))
        self.bias2 = nn.Parameter(torch.zeros(branches, node_dim))
        self.mix1 = nn.Parameter(torch.full((branches,), 1 / branches))
        self.mix2 = nn.Parameter(torch.full((branches,), 1 / branches))
        self.gate1 = nn.Linear(2 * node_dim, node_dim)
        self.gate2 = nn.Linear(2 * node_dim, node_dim)
        self.norm1 = nn.LayerNorm(node_dim) if norm else nn.Identity()
        self.norm2 = nn.LayerNorm(node_dim) if norm else nn.Identity()
        self.last_diagnostics = {}
        self.last_relations = []
        self.last_states = []

    def _gain(self, raw):
        return torch.sigmoid(raw)

    def _relation(self, q, k, gain):
        score = torch.matmul(q, k.transpose(1, 2)) / (self.node_dim ** 0.5)
        return torch.tanh(score / max(self.temperature, 1e-6)) * self._gain(gain)

    @staticmethod
    def _message(relation, source):
        return torch.einsum("bij,bid->bjd", relation, source)

    def _local(self, h, message, self_proj, msg_proj, bias, mix):
        b, n, d = h.shape
        a = self_proj(h).view(b, n, self.branches, d)
        m = msg_proj(message).view(b, n, self.branches, d)
        branches = torch.tanh(a + m + bias)
        weights = torch.softmax(mix, dim=0)
        return (branches * weights.view(1, 1, -1, 1)).sum(dim=2)

    def forward(self, h):
        expected = (2 * self.n_nodes, self.node_dim)
        if h.ndim != 3 or h.shape[1:] != expected:
            raise ValueError(f"expected (B,{expected[0]},{expected[1]}), got {tuple(h.shape)}")
        h1, h2 = h[:, :self.n_nodes], h[:, self.n_nodes:]
        self.last_relations, self.last_states = [], []
        changes, relation_changes = [], []
        prev_rel = None
        for _ in range(self.steps):
            q1, k1 = self.q1(h1), self.k1(h1)
            q2, k2 = self.q2(h2), self.k2(h2)
            r11 = self._relation(q1, k1, self.gain1_raw)
            r12 = self._relation(q1, k2, self.gain12_raw)
            r22 = (self._relation(q2, k2, self.gain2_raw)
                   if self.layer2_internal else torch.zeros_like(r11))
            r21 = (self._relation(q2, k1, self.gain21_raw)
                   if self.feedback else torch.zeros_like(r11))
            m1 = self._message(r11, h1)
            if self.feedback:
                m1 = m1 + self._message(r21, h2)
            m2 = self._message(r12, h1) + self._message(r22, h2)
            c1 = self._local(h1, m1, self.self1, self.msg1, self.bias1, self.mix1)
            c2 = self._local(h2, m2, self.self2, self.msg2, self.bias2, self.mix2)
            g1 = torch.sigmoid(self.gate1(torch.cat((h1, m1), dim=-1)))
            g2 = torch.sigmoid(self.gate2(torch.cat((h2, m2), dim=-1)))
            n1 = self.norm1((1 - g1) * h1 + g1 * c1)
            n2 = self.norm2((1 - g2) * h2 + g2 * c2)
            relation_pack = torch.stack((r11, r12, r22, r21), dim=0)
            state_pack = torch.cat((n1, n2), dim=1)
            if prev_rel is None:
                relation_changes.append(torch.zeros((), device=h.device, dtype=h.dtype))
            else:
                relation_changes.append((relation_pack - prev_rel).abs().mean())
            changes.append((state_pack - torch.cat((h1, h2), dim=1)).abs().mean())
            # relation_pack is [relation_type, batch, N, N].  Preserve every
            # relation type and retain only the first diagnostic sample.
            self.last_relations.append(relation_pack.detach()[:, :1])
            self.last_states.append(state_pack.detach()[:1])
            prev_rel = relation_pack
            h1, h2 = n1, n2
        final = torch.cat((h1, h2), dim=1)
        rel_stack = torch.stack(self.last_relations)
        self.last_diagnostics = {
            "state_change": torch.stack(changes).detach().cpu().tolist(),
            "relation_change": torch.stack(relation_changes).detach().cpu().tolist(),
            "state_abs_mean": [x.abs().mean().item() for x in self.last_states],
            "relation_abs_mean": [x.abs().mean().item() for x in self.last_relations],
            "r11_abs_mean": rel_stack[:, 0].abs().mean().item(),
            "r12_abs_mean": rel_stack[:, 1].abs().mean().item(),
            "r22_abs_mean": rel_stack[:, 2].abs().mean().item(),
            "r21_abs_mean": rel_stack[:, 3].abs().mean().item(),
            "feedback": self.feedback,
            "layer2_internal": self.layer2_internal,
        }
        return final

    def parameter_report(self):
        return {
            "parameters": sum(p.numel() for p in self.parameters()),
            "relation_entries_per_step": 3 * self.n_nodes * self.n_nodes +
            (self.n_nodes * self.n_nodes if self.feedback else 0),
            "relation_score_multiplies_per_step": 3 * self.n_nodes ** 2 * self.node_dim +
            (self.n_nodes ** 2 * self.node_dim if self.feedback else 0),
            "steps": self.steps,
        }


class HierarchicalDynamicNFMLP(nn.Module):
    def __init__(self, hidden=64, d_out=10, **field_cfg):
        super().__init__()
        if hidden != 64:
            raise ValueError("first version expects hidden=64")
        self.input = nn.Linear(784, hidden)
        self.field = HierarchicalDynamicNeuralField(**field_cfg)
        self.output = nn.Linear(hidden, d_out)

    def forward(self, x, return_hidden=False):
        h = self.input(x.flatten(1)).view(-1, 16, 4)
        h = self.field(h)
        y = self.output(h.flatten(1))
        return (y, h.flatten(1)) if return_hidden else y
