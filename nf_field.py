"""Discrete Neural Field MLP - PyTorch implementation of the Phi_NF operator.

Implements the architecture from <通用离散神经场MLP完整架构设计_v3.docx>:
replace the elementwise activation phi in a standard MLP with a discrete
neural field operator Phi_NF : R^d -> R^d, built from a 2D grid of coupled
spiking-like neurons (per-neuron threshold/delay/duration) that propagate
over K_t discrete time steps, read out via region pooling + temporal weighting.

Document section mapping:
  S (write)      Sec. 6-7
  F (propagate)  Sec. 8-10
  R (readout)    Sec. 11
  surrogate gradient (STE)   Sec. 14

Key implementation notes:
- The ring buffer (future-signal cache) is updated with *functional rebinding*
  (no in-place writes on gradient-requiring tensors), so autograd stays a DAG
  across the K_t sequential steps.
- Hard threshold forward, sigmoid-surrogate backward (straight-through
  estimator), same for the "is currently sending" indicator.
- Discrete params L (delay) and D (duration) are fixed in this phase.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _lattice(K_s: int) -> torch.Tensor:
    """Deterministic seed-bias lattice over (0,1)^2, kept off the edges."""
    side = int(math.sqrt(K_s))
    if side * side == K_s and side >= 2:
        g = torch.linspace(0.2, 0.8, side)
        c = torch.stack(torch.meshgrid(g, g, indexing="ij"), dim=-1).reshape(-1, 2)
    else:
        c = torch.linspace(0.2, 0.8, K_s).unsqueeze(1).expand(K_s, 2).clone()
    # _write applies sigmoid(q) to obtain normalized coordinates, so the
    # lattice coordinates must be stored in logit space.
    return torch.logit(c)  # (K_s, 2)


class DiscreteNeuralField(nn.Module):
    """The Phi_NF operator: R^d -> R^d."""

    def __init__(self, d, H=32, W=32, K_s=16, K_t=24, L_max=8, D_max=3,
                 R=12, tau=1.0, inject_scale=4.0, surr_scale=1.0,
                 read_mode="spike", eps_std=0.0, pool_mode="mean"):
        super().__init__()
        assert H * W % d == 0, "grid area must be divisible by d for region pooling"
        sd = math.isqrt(d)
        assert sd * sd == d, "d must be a perfect square for square region pooling"
        self.d, self.H, self.W = d, H, W
        self.K_s, self.K_t, self.L_max, self.D_max = K_s, K_t, L_max, D_max
        self.R, self.tau = R, tau
        self.inject_scale = inject_scale
        self.surr_scale = surr_scale  # amplifies STE gradient (training trick)
        assert read_mode in ("spike", "potential")
        self.read_mode = read_mode  # spike=m (doc), potential=h_arr (smooth path)
        assert pool_mode in ("mean", "max")
        self.pool_mode = pool_mode
        self.eps_std = eps_std        # doc §6: train-time seed-position perturbation
        self.N = H * W
        self._sd = sd
        self._rh, self._rw = H // sd, W // sd

        # --- seed generators (Sec. 6) ---
        self.A = nn.Parameter(torch.randn(K_s, 2, d) * 0.1)
        self.c = nn.Parameter(_lattice(K_s))
        self.w = nn.Parameter(torch.randn(K_s, d) * 0.1)
        self.b = nn.Parameter(torch.randn(K_s) * 0.1)

        # --- field parameters (Sec. 5) ---
        # T ~ U(0.15, 0.45): tuned so the injected signal (I ~ 0.6-1.2 with
        # inject_scale=4) fires a healthy fraction at t=0.
        self.T = nn.Parameter(torch.rand(H, W) * 0.3 + 0.15)
        # S = softplus(s_param), s_param ~ N(-0.2, 0.4) -> S ~ 0.3-0.6 so a
        # single active neighbor can push arrivals over T (wave sustains).
        self.s_param = nn.Parameter(torch.randn(H, W) * 0.4 - 0.2)

        # --- readout temporal weights (Sec. 11) ---
        self.eta = nn.Parameter(torch.zeros(R))

        # --- discrete params (Sec. 15: trained via local search) ---
        self.register_buffer("L", torch.randint(1, L_max + 1, (H, W)))
        self.register_buffer("D", torch.randint(1, D_max + 1, (H, W)))

        self._build_routing()

        # diagnostics (filled on demand)
        self._stats = None

    # ------------------------------------------------------------------ #
    # routing tables: flat source indices + target offset per direction    #
    # ------------------------------------------------------------------ #
    def _build_routing(self):
        H, W = self.H, self.W
        r = torch.arange(H)
        c = torch.arange(W)
        rr, cc = torch.meshgrid(r, c, indexing="ij")
        flat = rr * W + cc
        # direction "up": source has an up-neighbor (rows 1..H-1), target = src - W
        self.register_buffer("src_up", flat[1:, :].reshape(-1).long())
        self.register_buffer("off_up", torch.tensor(-W, dtype=torch.long))
        self.register_buffer("src_dn", flat[:-1, :].reshape(-1).long())
        self.register_buffer("off_dn", torch.tensor(W, dtype=torch.long))
        self.register_buffer("src_lf", flat[:, 1:].reshape(-1).long())
        self.register_buffer("off_lf", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("src_rt", flat[:, :-1].reshape(-1).long())
        self.register_buffer("off_rt", torch.tensor(1, dtype=torch.long))

    # ------------------------------------------------------------------ #
    # S: differentiable bilinear write (Sec. 6-7)                         #
    # ------------------------------------------------------------------ #
    def _write(self, h):
        """h: (B, d) -> I: (B, H, W), the t=0 stimulus field."""
        B = h.shape[0]
        q = torch.einsum("ksd,bd->bks", self.A, h) + self.c       # (B,K_s,2)
        if self.training and self.eps_std > 0:
            q = q + torch.randn_like(q) * self.eps_std            # doc §6: train-time perturbation
        grid = torch.tensor([self.H - 1, self.W - 1], device=h.device, dtype=h.dtype)
        p = grid * torch.sigmoid(q)                                # (B,K_s,2) continuous pos
        a = self.inject_scale * F.softplus(
            torch.einsum("kd,bd->bk", self.w, h) + self.b)  # (B,K_s)

        u, v = p[..., 0], p[..., 1]
        u0 = torch.clamp(torch.floor(u).long(), 0, self.H - 2)
        v0 = torch.clamp(torch.floor(v).long(), 0, self.W - 2)
        fu = (u - u0).clamp(0, 1)
        fv = (v - v0).clamp(0, 1)
        w00 = a * (1 - fu) * (1 - fv)
        w01 = a * (1 - fu) * fv
        w10 = a * fu * (1 - fv)
        w11 = a * fu * fv
        i00 = u0 * self.W + v0
        i01 = u0 * self.W + (v0 + 1)
        i10 = (u0 + 1) * self.W + v0
        i11 = (u0 + 1) * self.W + (v0 + 1)

        I = torch.zeros(B, self.N, device=h.device, dtype=h.dtype)
        for idx, wgt in [(i00, w00), (i01, w01), (i10, w10), (i11, w11)]:
            I = I.scatter_add(1, idx, wgt)
        return I.view(B, self.H, self.W)

    # ------------------------------------------------------------------ #
    # F: discrete propagation (Sec. 8-10)                                 #
    # ------------------------------------------------------------------ #
    def _propagate(self, I, inject=None):
        """Propagate K_t steps.

        I: (B,H,W) stimulus injected at t=0 only (stateless field use).
        inject: optional (B, K_t, H, W) per-step stimulus; when given, the field
        time becomes the sequence time (each step receives its own input).
        """
        B = I.shape[0]
        dev = I.device
        dt = I.dtype
        T = self.T
        S = F.softplus(self.s_param)
        Df = self.D.to(dt)                     # doc §5: duration buffer (float copy)
        L_flat = self.L.reshape(-1)            # doc §5: delay buffer

        # ring buffer of future-arriving signals; slot == arrival step
        inbox = torch.zeros(B, self.L_max, self.N, device=dev, dtype=dt)
        r = torch.zeros(B, self.H, self.W, device=dev, dtype=dt)  # remaining duration
        m_hist, h_hist = [], []
        z_means, a_means, m_means = [], [], []
        rho_sum = None                       # doc §16: mean firing rate (graph tensor)

        for t in range(self.K_t):
            slot = t % self.L_max
            h_arr = inbox[:, slot].view(B, self.H, self.W)   # arrivals due at t
            if inject is not None:
                h_arr = h_arr + inject[:, t]                 # sequence-time feeding
            elif t == 0:
                h_arr = h_arr + I                            # only t=0 injection
            h_hist.append(h_arr)

            # threshold activation with refractory gate (STE surrogate)
            z_hard = ((h_arr >= T) & (r == 0)).to(dt)
            z_surr = torch.sigmoid((h_arr - T) / self.tau) * (r == 0).to(dt)
            z = z_hard + self.surr_scale * (z_surr - z_surr.detach())

            r_tilde = z * Df + (1 - z) * r
            a_hard = (r_tilde > 0).to(dt)
            a_surr = torch.sigmoid(r_tilde / self.tau)
            a = a_hard + self.surr_scale * (a_surr - a_surr.detach())
            m = a * S                                          # (B,H,W) send activity
            m_hist.append(m)

            # ---- delay routing: per-neuron delay L_i -> future slot ----
            m_flat = m.reshape(B, -1)
            writes = torch.zeros(B, self.L_max * self.N, device=dev, dtype=dt)
            for _name, src, off in (("up", self.src_up, self.off_up),
                                    ("dn", self.src_dn, self.off_dn),
                                    ("lf", self.src_lf, self.off_lf),
                                    ("rt", self.src_rt, self.off_rt)):
                src_vals = m_flat[:, src]                       # (B, n_src)
                L_src = L_flat[src]                             # (n_src,)
                slot_i = (t + L_src) % self.L_max               # (n_src,)
                buf_idx = slot_i * self.N + (src + off)         # (n_src,)
                writes = writes.scatter_add(
                    1, buf_idx.unsqueeze(0).expand(B, -1), src_vals)

            # functional ring update: replace the just-read slot, add to the rest
            writes = writes.view(B, self.L_max, self.N)
            slots = [inbox[:, s] for s in range(self.L_max)]
            slots[slot] = writes[:, slot]
            for s in range(self.L_max):
                if s != slot:
                    slots[s] = slots[s] + writes[:, s]
            inbox = torch.stack(slots, dim=1)

            r = torch.clamp(r_tilde - 1, min=0)

            rho_sum = z.mean() if rho_sum is None else rho_sum + z.mean()
            z_means.append(z.mean().item())
            a_means.append(a.mean().item())
            m_means.append(m.mean().item())

        rho = rho_sum / self.K_t           # doc §16: (B,H,W) mean over time -> scalar
        return (torch.stack(m_hist, dim=0), torch.stack(h_hist, dim=0),
                (z_means, a_means, m_means), rho)

    # ------------------------------------------------------------------ #
    # R: region pooling + temporal weighting (Sec. 11)                    #
    # ------------------------------------------------------------------ #
    def _readout(self, sig):
        """sig: (K_t, B, H, W) -> v: (B, d)."""
        tail = sig[-self.R:]                                      # (R,B,H,W)
        sd, rh, rw = self._sd, self._rh, self._rw
        s_r = tail.reshape(-1, sd, rh, sd, rw)                    # (R*B, sd,rh,sd,rw)
        if self.pool_mode == "mean":
            p = s_r.mean(dim=(2, 4))
        else:
            p = s_r.amax(dim=(2, 4))
        p = p.reshape(self.R, -1, self.d)                         # (R,B,d)
        gamma = torch.softmax(self.eta, dim=0)                    # (R,)
        return torch.einsum("r,rbd->bd", gamma, p)                # (B,d)

    # ------------------------------------------------------------------ #
    def forward(self, h):
        I = self._write(h)
        m_hist, h_hist, (z_means, a_means, m_means), rho = self._propagate(I)
        sig = h_hist if self.read_mode == "potential" else m_hist
        v = self._readout(sig)
        self.last_rho = rho               # doc §16: mean firing rate (in autograd graph)
        if self._stats is not None:
            self._stats.update(
                z_means=z_means, a_means=a_means, m_means=m_means,
                v_mean=v.mean().item(), v_pos_frac=(v > 0).float().mean().item(),
            )
        return v

    def forward_grid(self, I):
        """Run propagation/readout from a sample-specific (B,H,W) grid."""
        if I.ndim != 3 or I.shape[1:] != (self.H, self.W):
            raise ValueError(f"expected input grid (B,{self.H},{self.W}), got {tuple(I.shape)}")
        m_hist, h_hist, (z_means, a_means, m_means), rho = self._propagate(I)
        sig = h_hist if self.read_mode == "potential" else m_hist
        v = self._readout(sig)
        self.last_rho = rho
        if self._stats is not None:
            self._stats.update(z_means=z_means, a_means=a_means, m_means=m_means,
                               v_mean=v.mean().item(), v_pos_frac=(v > 0).float().mean().item())
        return v

    # ------------------------------------------------------------------ #
    def forward_seq(self, h_seq):
        """Sequence input: h_seq (B, T, d) -> v (B, d).

        Requires T == self.K_t. Each step's hidden vector is turned into a
        seed-field (bilinear write) injected into the field's OWN time, so the
        field's delay/duration dynamics operate across sequence time.
        """
        B, T, d = h_seq.shape
        assert T == self.K_t, f"forward_seq requires T==K_t, got T={T}, K_t={self.K_t}"
        inject = torch.stack(
            [self._write(h_seq[:, t]) for t in range(T)], dim=1)   # (B,T,H,W)
        I = torch.zeros(B, self.H, self.W, device=h_seq.device, dtype=h_seq.dtype)
        m_hist, h_hist, (z_means, a_means, m_means), rho = self._propagate(I, inject=inject)
        sig = h_hist if self.read_mode == "potential" else m_hist
        v = self._readout(sig)
        self.last_rho = rho
        if self._stats is not None:
            self._stats.update(
                z_means=z_means, a_means=a_means, m_means=m_means,
                v_mean=v.mean().item(), v_pos_frac=(v > 0).float().mean().item(),
            )
        return v

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def step_discrete(self, probe_loss, n_neurons=32):
        """Greedy local search on L_i, D_i (doc §15).

        probe_loss: callable returning a scalar loss (already detached) for a
        fixed probe batch. For a random subset of neurons, test L_i +/- 1 and
        D_i +/- 1 within bounds; accept a candidate iff it lowers probe_loss.
        """
        N = self.N
        dev = self.L.device
        idx = torch.randint(0, N, (n_neurons,), device=dev)
        L = self.L.reshape(-1)
        D = self.D.reshape(-1)
        base = probe_loss()
        for i in idx.tolist():
            li = int(L[i].item())
            for cand in (li - 1, li + 1):
                if 1 <= cand <= self.L_max:
                    L[i] = cand
                    if probe_loss() < base:
                        base = probe_loss()
                        break
                    L[i] = li
            di = int(D[i].item())
            for cand in (di - 1, di + 1):
                if 1 <= cand <= self.D_max:
                    D[i] = cand
                    if probe_loss() < base:
                        base = probe_loss()
                        break
                    D[i] = di

    def enable_stats(self):
        self._stats = {}

    def disable_stats(self):
        self._stats = None

    @property
    def S(self):
        return F.softplus(self.s_param)


class NFMLPBlock(nn.Module):
    """A single NF-MLP block: W_up -> Phi_NF -> W_down."""

    def __init__(self, d_in, d, d_out, field_cfg):
        super().__init__()
        self.input_scale = field_cfg.get("input_scale", 3.0)
        self.W_up = nn.Linear(d_in, d)
        self.field = DiscreteNeuralField(d, **field_cfg)
        self.W_down = nn.Linear(d, d_out)
        nn.init.uniform_(self.W_down.weight, -0.05, 0.05)
        nn.init.uniform_(self.W_down.bias, -0.05, 0.05)

    def forward(self, x):
        h = self.W_up(x)
        v = self.field(h)
        return self.W_down(v)


class NFGridMLPBlock(nn.Module):
    """Diagnostic block that maps each sample directly to the field grid."""

    def __init__(self, d_in, d, d_out, field_cfg):
        super().__init__()
        self.grid_scale = 0.25
        H, W = field_cfg.get("H", 32), field_cfg.get("W", 32)
        self.W_grid = nn.Linear(d_in, H * W)
        self.field = DiscreteNeuralField(d, **field_cfg)
        self.W_down = nn.Linear(d, d_out)
        nn.init.uniform_(self.W_down.weight, -0.05, 0.05)
        nn.init.uniform_(self.W_down.bias, -0.05, 0.05)

    def forward(self, x):
        I = (self.grid_scale * F.softplus(self.W_grid(x))).view(
            x.shape[0], self.field.H, self.field.W)
        v = self.field.forward_grid(I)
        return self.W_down(v)


class NFCNNBlock(nn.Module):
    """Spatial NF block: preserve image locality before field propagation."""

    def __init__(self, d, d_out, field_cfg):
        super().__init__()
        self.input_scale = 0.5
        H, W = field_cfg.get("H", 32), field_cfg.get("W", 32)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1),
        )
        nn.init.normal_(self.conv[0].weight, std=0.05)
        nn.init.constant_(self.conv[0].bias, 0.0)
        nn.init.normal_(self.conv[2].weight, std=0.05)
        nn.init.constant_(self.conv[2].bias, -1.0)
        self.field = DiscreteNeuralField(d, **field_cfg)
        self.W_down = nn.Linear(d, d_out)
        nn.init.uniform_(self.W_down.weight, -0.05, 0.05)
        nn.init.uniform_(self.W_down.bias, -0.05, 0.05)
        self.H, self.W = H, W

    def forward(self, x):
        grid = self.conv(x)
        if grid.shape[-2:] != (self.H, self.W):
            grid = F.interpolate(grid, size=(self.H, self.W), mode="bilinear", align_corners=False)
        I = self.input_scale * F.softplus(grid[:, 0])
        v = self.field.forward_grid(I)
        return self.W_down(v)


class NFPoolCNNBlock(nn.Module):
    """Convolution + pooling front-end followed by the NF field."""

    def __init__(self, d, d_out, field_cfg):
        super().__init__()
        H, W = field_cfg.get("H", 32), field_cfg.get("W", 32)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
        )
        nn.init.constant_(self.conv[3].bias, 0.12)
        self.field = DiscreteNeuralField(d, **field_cfg)
        self.W_down = nn.Linear(d, d_out)
        nn.init.uniform_(self.W_down.weight, -0.05, 0.05)
        nn.init.uniform_(self.W_down.bias, -0.05, 0.05)
        self.H, self.W = H, W
        self.input_scale = 2.0

    def forward(self, x):
        grid = self.conv(x)
        grid = F.interpolate(grid, size=(self.H, self.W), mode="bilinear", align_corners=False)
        I = self.input_scale * F.relu(grid[:, 0])
        v = self.field.forward_grid(I)
        return self.W_down(v)


class DirectionalRectNeuralField(nn.Module):
    """v4 directional rectangular neural field, Phi_RF: R^d -> R^d."""

    def __init__(self, d, W=16, tau_a=0.2, tau_p=1.0, residual_alpha=0.0,
                 gain_init=1.0, train_gain=True, energy_mode="linear",
                 energy_scale=1.0, route_mode="all", threshold_init=0.5):
        super().__init__()
        self.d, self.W = d, W
        self.active_W = W
        self.tau_a, self.tau_p = tau_a, tau_p
        self.residual_alpha = residual_alpha
        self.hard_gate = True
        self._stats = None
        self.gain_init = gain_init
        if energy_mode not in ("linear", "log", "softlog", "tanh"):
            raise ValueError(energy_mode)
        self.energy_mode, self.energy_scale = energy_mode, energy_scale
        if route_mode not in ("all", "diagonal", "parallel_ru", "parallel_rud", "receiver_mix", "inbound_kernel", "energy_attention", "inbound_5", "inbound_5_colmean", "inbound_5_colattr", "inbound_full"):
            raise ValueError(route_mode)
        self.route_mode = route_mode

        # T = softplus(theta), initialized near the v4 reference T ~= 0.5.
        theta0 = math.log(math.expm1(threshold_init))
        self.theta = nn.Parameter(torch.full((W, d), theta0))
        self.kernel_raw = None
        self.column_attr = None
        self.full_raw = None
        if route_mode == "inbound_kernel":
            # One bounded local kernel per receiving neuron.  Its three
            # coefficients jointly encode direction and strength.
            k0 = torch.tensor([0.25, 0.50, 0.25]).view(1, 1, 3).expand(W, d, 3)
            self.kernel_raw = nn.Parameter(torch.logit(k0))
            self.g_raw = None
            self.Q = None
        elif route_mode == "energy_attention":
            # Shared energy evaluator: no direction-specific bias.  The same
            # two scalars score every incoming signal by magnitude and sign.
            self.energy_score_gain = nn.Parameter(torch.tensor(1.0))
            self.energy_score_sign = nn.Parameter(torch.tensor(0.0))
            self.g_raw = None
            self.Q = None
        elif route_mode in ("inbound_5", "inbound_5_colmean", "inbound_5_colattr"):
            k0 = torch.tensor([0.10, 0.20, 0.40, 0.20, 0.10]).view(1, 1, 5).expand(W, d, 5)
            self.kernel_raw = nn.Parameter(torch.logit(k0))
            if route_mode == "inbound_5_colattr":
                self.column_attr = nn.Parameter(torch.zeros(W, d, 2))
            self.g_raw = None
            self.Q = None
        elif route_mode == "inbound_full":
            self.full_raw = nn.Parameter(torch.zeros(W, d, d))
            self.g_raw = None
            self.Q = None
        else:
            # G = 2 * sigmoid(g_raw), so g_raw=0 gives G=1.
            g0 = math.log(gain_init / (2.0 - gain_init))
            self.g_raw = nn.Parameter(torch.full((W, d), g0))
            self.g_raw.requires_grad_(train_gain)
            self.Q = nn.Parameter(torch.randn(W, d, 3) * 0.01)

        template = torch.tensor([[1., 1., 1.],
                                 [1., 1., 0.],
                                 [0., 1., 1.]])
        self.register_buffer("template", template)

    def _route(self, q):
        """Hard direction choice with softmax STE and boundary masking."""
        if self.route_mode in ("parallel_ru", "parallel_rud"):
            # Parallel fan-out: every allowed direction is present. Route
            # indices are [down, right, up]. No branch is hard-selected.
            ids = (1, 2) if self.route_mode == "parallel_ru" else (0, 1, 2)
            weights = torch.softmax(q[..., ids] / self.tau_p, dim=-1)
            route = torch.zeros_like(q)
            route[..., list(ids)] = weights
        elif self.route_mode == "diagonal":
            # Keep only up+right and right+down.  The all-three route is
            # removed to avoid a source splitting into three chains.
            pi2 = torch.softmax(q[..., (0, 2)] / self.tau_p, dim=-1)
            hard2 = F.one_hot(pi2.argmax(dim=-1), num_classes=2).to(pi2.dtype)
            z2 = hard2 + (pi2 - pi2.detach())
            route = torch.zeros_like(q)
            route[..., 0] = z2[..., 0]
            route[..., 1] = 1.0
            route[..., 2] = z2[..., 1]
        else:
            pi = torch.softmax(q / self.tau_p, dim=-1)
            hard = F.one_hot(pi.argmax(dim=-1), num_classes=3).to(pi.dtype)
            z = hard + (pi - pi.detach())
            route = z @ self.template
        valid = torch.ones_like(route)
        valid[-1, 0] = 0.0       # source at bottom cannot shift down
        valid[0, 2] = 0.0        # source at top cannot shift up
        route = route * valid
        return route / (route.sum(dim=-1, keepdim=True) + 1e-6)

    @staticmethod
    def _shift_down(x):
        out = torch.zeros_like(x)
        out[:, 1:] = x[:, :-1]
        return out

    @staticmethod
    def _shift_up(x):
        out = torch.zeros_like(x)
        out[:, :-1] = x[:, 1:]
        return out

    @staticmethod
    def _shift(x, delta):
        out = torch.zeros_like(x)
        if delta > 0:
            out[:, delta:] = x[:, :-delta]
        elif delta < 0:
            out[:, :delta] = x[:, -delta:]
        else:
            out.copy_(x)
        return out

    def forward(self, h):
        if h.ndim != 2 or h.shape[-1] != self.d:
            raise ValueError(f"expected (B,{self.d}), got {tuple(h.shape)}")
        e = h
        z_means, a_means, e_abs_means = [], [], []
        e_energy = []
        for c in range(self.active_W):
            T = F.softplus(self.theta[c])
            G = (torch.sigmoid(self.g_raw[c]) * 2.0
                 if self.route_mode not in ("inbound_kernel", "energy_attention",
                                             "inbound_5", "inbound_5_colmean",
                                             "inbound_5_colattr", "inbound_full")
                 else 1.0)
            a_soft = torch.sigmoid((e.abs() - T) / self.tau_a)
            a_hard = (e.abs() >= T).to(e.dtype)
            a = (a_hard + (a_soft - a_soft.detach())) if self.hard_gate else a_soft
            if self.energy_mode == "linear":
                e_tx = e
            elif self.energy_mode == "log":
                e_tx = e.sign() * self.energy_scale * torch.log1p(e.abs() / self.energy_scale)
            elif self.energy_mode == "softlog":
                a_e = e.abs()
                excess = torch.relu(a_e - self.energy_scale)
                compressed = self.energy_scale + self.energy_scale * torch.log1p(
                    excess / self.energy_scale)
                e_tx = e.sign() * torch.where(a_e <= self.energy_scale, a_e, compressed)
            else:
                e_tx = self.energy_scale * torch.tanh(e / self.energy_scale)
            m = a * G * e_tx
            z_means.append(a_hard.mean().item())
            a_means.append((m != 0).to(e.dtype).mean().item())
            e_abs_means.append(e.abs().mean().item())
            e_energy.append(e.abs().mean())

            if self.route_mode == "energy_attention":
                # Score each incoming signal from its own energy.  The score
                # function is shared across directions and positions.
                from_below = self._shift_down(m)
                from_left = m
                from_above = self._shift_up(m)
                incoming = torch.stack((from_below, from_left, from_above), dim=-1)
                scale = max(float(self.energy_scale), 1e-6)
                magnitude = torch.log1p(incoming.abs() / scale)
                signed = torch.tanh(incoming / scale)
                scores = (self.energy_score_gain * magnitude +
                          self.energy_score_sign * signed) / self.tau_p
                valid = torch.ones((self.d, 3), device=m.device, dtype=m.dtype)
                valid[0, 0] = 0.0
                valid[-1, 2] = 0.0
                scores = scores.masked_fill(valid.unsqueeze(0) == 0, -1e9)
                weights = torch.softmax(scores, dim=-1)
                e = (incoming * weights).sum(dim=-1)
            elif self.route_mode in ("inbound_5", "inbound_5_colmean", "inbound_5_colattr"):
                offsets = (-2, -1, 0, 1, 2)
                incoming = torch.stack(tuple(self._shift(m, dlt) for dlt in offsets), dim=-1)
                weights = torch.softmax(self.kernel_raw[c] / self.tau_p, dim=-1)
                valid = torch.ones((self.d, 5), device=m.device, dtype=m.dtype)
                for i in range(self.d):
                    for k, dlt in enumerate(offsets):
                        if i - dlt < 0 or i - dlt >= self.d:
                            valid[i, k] = 0.0
                weights = weights * valid
                weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)
                e = (incoming * weights.unsqueeze(0)).sum(dim=-1)
                if self.route_mode == "inbound_5_colmean":
                    e = e + 0.10 * m.mean(dim=-1, keepdim=True)
                elif self.route_mode == "inbound_5_colattr":
                    attr = torch.stack((m.mean(dim=-1), m.abs().mean(dim=-1)), dim=-1)
                    e = e + attr @ self.column_attr[c].transpose(0, 1)
            elif self.route_mode == "inbound_full":
                weights = torch.softmax(self.full_raw[c] / self.tau_p, dim=-1)
                e = m @ weights.transpose(0, 1)
            elif self.route_mode == "inbound_kernel":
                # Receiver-centric local propagation.  The target receives
                # all three valid neighbors; the same kernel determines both
                # which source matters and how strongly it is transmitted.
                weights = torch.sigmoid(self.kernel_raw[c])
                valid = torch.ones_like(weights)
                valid[0, 0] = 0.0
                valid[-1, 2] = 0.0
                from_below = self._shift_down(m)
                from_left = m
                from_above = self._shift_up(m)
                e = (from_below * weights[:, 0] +
                     from_left * weights[:, 1] +
                     from_above * weights[:, 2])
            elif self.route_mode == "receiver_mix":
                # Receiver-centric propagation.  The target neuron receives
                # all valid neighbors, then learns their relative importance.
                # Q columns are [from-left-below, from-left, from-left-above].
                weights = torch.softmax(self.Q[c] / self.tau_p, dim=-1)
                valid = torch.ones_like(weights)
                valid[0, 0] = 0.0       # target top cannot receive from below
                valid[-1, 2] = 0.0      # target bottom cannot receive from above
                weights = weights * valid
                weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)
                from_below = self._shift_down(m)
                from_left = m
                from_above = self._shift_up(m)
                e = (from_below * weights[:, 0] +
                     from_left * weights[:, 1] +
                     from_above * weights[:, 2])
            else:
                route = self._route(self.Q[c])
                up = m * route[:, 0]
                right = m * route[:, 1]
                down = m * route[:, 2]
                e = self._shift_down(up) + right + self._shift_up(down)

        out = h + self.residual_alpha * e if self.residual_alpha else e
        self.relay_energy_first = e_energy[0]
        self.relay_energy_last = e_energy[-1]
        self.relay_ratio = self.relay_energy_last / (self.relay_energy_first + 1e-6)
        if self._stats is not None:
            self._stats.update(z_means=z_means, a_means=a_means,
                               e_abs_means=e_abs_means,
                               v_pos_frac=(out > 0).float().mean().item())
        return out

    def set_active_W(self, width):
        if not 1 <= width <= self.W:
            raise ValueError(f"active width must be in [1,{self.W}], got {width}")
        self.active_W = int(width)

    def set_hard_gate(self, enabled):
        self.hard_gate = bool(enabled)

    def enable_stats(self):
        self._stats = {}

    def disable_stats(self):
        self._stats = None


class RectNFMLPBlock(nn.Module):
    """MLP block using the v4 directional rectangular field."""

    def __init__(self, d_in, d, d_out, field_cfg):
        super().__init__()
        self.input_scale = 3.0
        self.W_up = nn.Linear(d_in, d)
        self.field = DirectionalRectNeuralField(d, **field_cfg)
        self.W_down = nn.Linear(d, d_out)
        nn.init.uniform_(self.W_down.weight, -0.05, 0.05)
        nn.init.uniform_(self.W_down.bias, -0.05, 0.05)

    def forward(self, x):
        return self.W_down(self.field(self.input_scale * self.W_up(x)))

    @torch.no_grad()
    def calibrate_input_scale(self, x, target_energy=0.02, min_scale=0.25,
                              max_scale=32.0, steps=128):
        """Choose a fixed pre-training scale that reaches the right edge.

        This is deliberately a one-time calibration of the whole input energy,
        not a trainable relay/gain term.  It uses only a training probe and the
        field's initial parameters.  The smallest scale whose last column has
        the requested mean absolute energy is selected, so the field starts
        coarse but alive at the right edge. A scan is used instead of binary
        search because hard thresholding and discrete routing make activity
        non-monotone.
        """
        old_scale = self.input_scale
        old_training = self.training
        self.eval()

        def last_activity(scale):
            self.input_scale = float(scale)
            self.field.enable_stats()
            _ = self(x)
            value = self.field._stats["z_means"][-1]
            self.field.disable_stats()
            return float(value)

        chosen = float(max_scale)
        achieved = 0.0
        for scale in torch.linspace(min_scale, max_scale, steps).tolist():
            value = last_activity(scale)
            if float(self.field.relay_energy_last) >= target_energy:
                chosen, achieved = float(scale), value
                break
        else:
            achieved = last_activity(chosen)

        self.input_scale = chosen
        self.field.enable_stats()
        _ = self(x)
        energy_first = float(self.field.relay_energy_first)
        energy_last = float(self.field.relay_energy_last)
        self.field.disable_stats()
        self.train(old_training)
        return chosen, achieved, energy_first, energy_last


class Geo10RectNFBlock(nn.Module):
    """Rectangular NF with ten fixed geometric readout points.

    The field keeps the same learned input projection, but there is no learned
    W_down and no internal route supervision. Ten evenly spaced neurons are
    treated as the ten class points; classification sees only their energy.
    """

    def __init__(self, d_in, height, field_cfg):
        super().__init__()
        self.W_up = nn.Linear(d_in, height)
        self.input_scale = 3.0
        self.field = DirectionalRectNeuralField(height, **field_cfg)
        anchors = torch.linspace(0, height - 1, 10).round().long()
        self.register_buffer("anchors", anchors)

    def forward(self, x):
        h = self.input_scale * self.W_up(x)
        e = self.field(h)
        return e[:, self.anchors].abs()
