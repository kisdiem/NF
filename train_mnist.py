"""NF-MLP vs standard MLP baselines on MNIST (sanity check).

Compares three model tiers on identical data / optimizer / epochs:
  1. original-MNIST reference : classic MLP 784->256->10 ReLU  (~98%)
  2. same-width fair baselines: MLP 784->64->10 with ReLU / GELU
  3. NF-MLP                  : W_up(784->64) + Phi_NF + W_down(64->10)

Diagnostics for the NF path (first batch of each epoch): activation rate,
send rate, per-step mean activity timeline, and parameter grad norms.
"""
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from nf_field import NFMLPBlock, NFGridMLPBlock, NFCNNBlock, NFPoolCNNBlock, RectNFMLPBlock

DEFAULT_FIELD = dict(H=32, W=32, K_s=16, K_t=24, L_max=8, D_max=3, R=12,
                     tau=1.0, inject_scale=4.0, surr_scale=1.0,
                     read_mode="spike", eps_std=0.0)


def build_model(name, field_cfg=None):
    if name == "nf":
        cfg = {**DEFAULT_FIELD, **(field_cfg or {})}
        return nn.Sequential(nn.Flatten(), NFMLPBlock(784, 64, 10, cfg))
    if name == "rect_nf":
        cfg = {"W": 16, "tau_a": 0.2, "tau_p": 1.0, "residual_alpha": 1.0,
               "route_mode": "all"}
        cfg.update({k: v for k, v in (field_cfg or {}).items()
                    if k in ("W", "tau_a", "tau_p", "residual_alpha", "gain_init", "train_gain",
                              "energy_mode", "energy_scale")})
        cfg["route_mode"] = (field_cfg or {}).get("route_mode", "all")
        cfg["threshold_init"] = (field_cfg or {}).get("threshold_init", 0.5)
        height = (field_cfg or {}).get("height", 64)
        return nn.Sequential(nn.Flatten(), RectNFMLPBlock(784, height, 10, cfg))
    if name == "nf_grid":
        cfg = {**DEFAULT_FIELD, **(field_cfg or {})}
        return nn.Sequential(nn.Flatten(), NFGridMLPBlock(784, 64, 10, cfg))
    if name == "nf_cnn":
        cfg = {**DEFAULT_FIELD, **(field_cfg or {})}
        return nn.Sequential(nn.Identity(), NFCNNBlock(64, 10, cfg))
    if name == "nf_cnn_pool":
        cfg = {**DEFAULT_FIELD, **(field_cfg or {})}
        return nn.Sequential(nn.Identity(), NFPoolCNNBlock(64, 10, cfg))
    if name == "cnn":
        return nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 10),
        )
    if name == "relu64":
        return nn.Sequential(nn.Flatten(), nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 10))
    if name == "linear64":
        return nn.Sequential(nn.Flatten(), nn.Linear(784, 64), nn.Linear(64, 10))
    if name == "gelu64":
        return nn.Sequential(nn.Flatten(), nn.Linear(784, 64), nn.GELU(), nn.Linear(64, 10))
    if name == "mlp256":
        return nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    raise ValueError(name)


def load_data(root, batch_size, subset):
    tf = transforms.ToTensor()
    train_ds = datasets.MNIST(root=root, train=True, download=True, transform=tf)
    test_ds = datasets.MNIST(root=root, train=False, download=True, transform=tf)
    if subset:
        train_ds = Subset(train_ds, torch.arange(subset))
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(test_ds, batch_size=1024, shuffle=False))


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total


def grad_norms(model):
    out = {}
    for n, p in model.named_parameters():
        if p.grad is not None:
            out[n] = p.grad.norm().item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nf",
                    choices=["nf", "rect_nf", "nf_grid", "nf_cnn", "nf_cnn_pool", "cnn", "linear64", "relu64", "gelu64", "mlp256"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--subset", type=int, default=0, help="0 = full 60k train set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", default="data/mnist")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--inject-scale", type=float, default=None, dest="inject_scale")
    ap.add_argument("--surr-scale", type=float, default=None, dest="surr_scale")
    ap.add_argument("--Kt", type=int, default=None, dest="K_t")
    ap.add_argument("--Ks", type=int, default=None, dest="K_s",
                    help="number of differentiable seed generators")
    ap.add_argument("--R", type=int, default=None,
                    help="number of final field steps used by readout")
    ap.add_argument("--read-mode", default=None, choices=["spike", "potential"])
    ap.add_argument("--pool", default=None, choices=["mean", "max"], dest="pool_mode")
    ap.add_argument("--rect-W", type=int, default=None, dest="rect_W",
                    help="v4 rectangular field width")
    ap.add_argument("--rect-tau-a", type=float, default=None, dest="rect_tau_a")
    ap.add_argument("--rect-tau-p", type=float, default=None, dest="rect_tau_p")
    ap.add_argument("--rect-residual", type=float, default=None, dest="rect_residual")
    ap.add_argument("--rect-gain", type=float, default=None, dest="rect_gain")
    ap.add_argument("--rect-freeze-gain", action="store_true", dest="rect_freeze_gain")
    ap.add_argument("--rect-energy", choices=["linear", "log", "softlog", "tanh"],
                    default=None, dest="rect_energy")
    ap.add_argument("--rect-energy-scale", type=float, default=None,
                    dest="rect_energy_scale")
    ap.add_argument("--rect-route",
                    choices=["all", "diagonal", "parallel_ru", "parallel_rud", "receiver_mix", "inbound_kernel", "energy_attention", "inbound_5", "inbound_5_colmean", "inbound_5_colattr", "inbound_full"],
                    default="all",
                    dest="rect_route",
                    help="parallel modes retain all allowed propagation branches")
    ap.add_argument("--rect-height", type=int, default=64, dest="rect_height",
                    help="number of neurons in the rectangular field height")
    ap.add_argument("--rect-threshold", type=float, default=0.5,
                    dest="rect_threshold", help="initial threshold T")
    ap.add_argument("--rect-gain-lr", type=float, default=None,
                    dest="rect_gain_lr",
                    help="separate learning rate for per-neuron gain G")
    ap.add_argument("--rect-kernel-lr", type=float, default=None,
                    dest="rect_kernel_lr",
                    help="separate learning rate for inbound propagation kernels")
    ap.add_argument("--rect-soft-warmup", type=int, default=0,
                    dest="rect_soft_warmup",
                    help="number of initial epochs using a soft threshold gate")
    ap.add_argument("--rect-curriculum", action="store_true",
                    help="grow v4 field width after the current right edge is active")
    ap.add_argument("--rect-start-W", type=int, default=1, dest="rect_start_W")
    ap.add_argument("--rect-grow-threshold", type=float, default=0.05,
                    dest="rect_grow_threshold")
    ap.add_argument("--rect-calibrate-energy", action="store_true",
                    help="before training, calibrate fixed input energy so the last field column is active")
    ap.add_argument("--rect-calibrate-target", type=float, default=0.02,
                    dest="rect_calibrate_target",
                    help="minimum last-column mean absolute energy for pre-training calibration")
    ap.add_argument("--lambda-relay", type=float, default=0.0,
                    help="penalty for relay energy ratio outside [0.2, 2.0]")
    ap.add_argument("--eps-std", type=float, default=None, dest="eps_std",
                    help="doc §6: train-time seed-position noise std")
    ap.add_argument("--lambd-act", type=float, default=0.0,
                    help="doc §16: activation-constraint weight (0 = off)")
    ap.add_argument("--rho0", type=float, default=0.1,
                    help="doc §16: target mean firing rate")
    ap.add_argument("--ld-every", type=int, default=0,
                    help="doc §15: run L/D local search every N batches (0 = off)")
    ap.add_argument("--ld-n", type=int, default=32,
                    help="doc §15: neurons to test per local-search step")
    ap.add_argument("--probe-size", type=int, default=32,
                    help="probe batch size for L/D local search")
    args = ap.parse_args()

    field_cfg = {k: v for k, v in {
        "tau": args.tau, "inject_scale": args.inject_scale,
        "surr_scale": args.surr_scale, "K_t": args.K_t,
        "K_s": args.K_s, "R": args.R,
        "pool_mode": args.pool_mode,
        "read_mode": args.read_mode, "eps_std": args.eps_std,
    }.items() if v is not None}
    rect_cfg = {k: v for k, v in {
        "W": args.rect_W, "tau_a": args.rect_tau_a,
        "tau_p": args.rect_tau_p, "residual_alpha": args.rect_residual,
        "gain_init": args.rect_gain,
        "train_gain": False if args.rect_freeze_gain else None,
        "energy_mode": args.rect_energy,
        "energy_scale": args.rect_energy_scale,
        "route_mode": args.rect_route,
        "height": args.rect_height,
        "threshold_init": args.rect_threshold,
    }.items() if v is not None}

    torch.manual_seed(args.seed)
    device = args.device
    print(f"[{args.model}] device={device} epochs={args.epochs} lr={args.lr} "
          f"batch={args.batch} subset={args.subset or 'full'}")

    train_loader, test_loader = load_data(args.data_root, args.batch, args.subset)
    model = build_model(args.model, rect_cfg if args.model == "rect_nf" else field_cfg).to(device)

    if args.model == "rect_nf" and args.rect_calibrate_energy:
        # Calibration uses a training batch only; no validation/test signal is
        # involved, and the resulting scale is fixed before the optimizer runs.
        cx, _ = next(iter(train_loader))
        cx = cx.to(device).flatten(1)
        block = model[1]
        scale, activity, e0, eend = block.calibrate_input_scale(
            cx, target_energy=args.rect_calibrate_target)
        print(f"  energy_calibration: input_scale={scale:.4f} "
              f"last_activity={activity:.3f} e0={e0:.4f} eEnd={eend:.4f}")
    if args.model == "rect_nf" and (args.rect_gain_lr is not None or args.rect_kernel_lr is not None):
        special_names = []
        special_lrs = []
        if args.rect_gain_lr is not None:
            special_names.append("field.g_raw")
            special_lrs.append(args.rect_gain_lr)
        if args.rect_kernel_lr is not None:
            for name in ("field.kernel_raw", "field.column_attr", "field.full_raw"):
                special_names.append(name)
                special_lrs.append(args.rect_kernel_lr)
        special_params = []
        for needle in special_names:
            special_params += [p for n, p in model.named_parameters() if needle in n]
        special_ids = {id(p) for p in special_params}
        base_params = [p for p in model.parameters() if id(p) not in special_ids]
        groups = [{"params": base_params, "lr": args.lr}]
        for needle, lr in zip(special_names, special_lrs):
            params = [p for n, p in model.named_parameters() if needle in n]
            if params:
                groups.append({"params": params, "lr": lr})
        opt = torch.optim.Adam([
            *groups,
        ], weight_decay=1e-4)
        print(f"  optimizer: base_lr={args.lr} "
              f"gain_lr={args.rect_gain_lr} kernel_lr={args.rect_kernel_lr}")
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.model in ("nf", "rect_nf", "nf_grid", "nf_cnn", "nf_cnn_pool"):
        field = model[1].field
        if args.model == "rect_nf" and args.rect_curriculum:
            field.set_active_W(args.rect_start_W)

    # fixed probe batch for doc §15 L/D local search (evaluated under no_grad)
    probe = None
    if args.model in ("nf", "nf_cnn", "nf_cnn_pool") and args.ld_every > 0:
        px, py = next(iter(test_loader))
        probe = (px[:args.probe_size].to(device), py[:args.probe_size].to(device))

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        if args.model == "rect_nf" and args.rect_soft_warmup:
            field.set_hard_gate(ep >= args.rect_soft_warmup)
            print(f"  threshold_gate: {'hard' if field.hard_gate else 'soft'}")
        run_loss = run_iters = 0
        grow_ready = False
        for bi, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            if args.model in ("nf", "rect_nf", "nf_grid", "nf_cnn", "nf_cnn_pool"):
                field.enable_stats()
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            if args.model == "rect_nf" and args.lambda_relay > 0:
                ratio = field.relay_ratio
                relay_penalty = F.relu(0.2 - ratio).pow(2) + F.relu(ratio - 2.0).pow(2)
                loss = loss + args.lambda_relay * relay_penalty
            if args.model in ("nf", "rect_nf", "nf_grid", "nf_cnn", "nf_cnn_pool") and args.lambd_act > 0:
                # doc §16: activation constraint on mean firing rate
                loss = loss + args.lambd_act * (field.last_rho - args.rho0) ** 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if args.model in ("nf", "nf_cnn") and args.ld_every > 0 and (bi + 1) % args.ld_every == 0:
                # doc §15: greedy discrete local search on L_i, D_i
                model.eval()
                px, py = probe
                field.step_discrete(
                    lambda: F.cross_entropy(model(px), py).item(), n_neurons=args.ld_n)
                model.train()

            run_loss += loss.item() * y.numel()
            run_iters += y.numel()

            if args.model in ("nf", "rect_nf", "nf_grid", "nf_cnn", "nf_cnn_pool") and bi == 0:
                st = field._stats
                zm, am = st["z_means"], st["a_means"]
                print(f"  diag[ep{ep}] z_mean={sum(zm)/len(zm):.3f} "
                      f"a_mean={sum(am)/len(am):.3f} "
                      f"z_t0={zm[0]:.3f} z_tEnd={zm[-1]:.3f} "
                      f"v_pos_frac={st['v_pos_frac']:.2f}")
                if "e_abs_means" in st:
                    em = st["e_abs_means"]
                    print(f"  relay_abs: e0={em[0]:.4f} eEnd={em[-1]:.4f} "
                          f"ratio={em[-1] / (em[0] + 1e-8):.3f}")
                    grow_ready = em[-1] >= args.rect_grow_threshold
                gn = grad_norms(model)
                key = [k for k in gn if "field.T" in k or "field.s_param" in k
                       or k.endswith("A") or k.endswith("W_up.weight")]
                print("  grads:", {k: round(gn[k], 4) for k in key if k in gn})
                if args.model == "rect_nf":
                    if field.g_raw is not None:
                        gain = (2.0 * torch.sigmoid(field.g_raw)).detach()
                        gg = gn.get("1.field.g_raw", 0.0)
                        print(f"  gain: mean={gain.mean().item():.4f} "
                              f"min={gain.min().item():.4f} max={gain.max().item():.4f} "
                              f"grad={gg:.6f}")
                    elif field.kernel_raw is not None:
                        kernel = torch.sigmoid(field.kernel_raw).detach()
                        kg = gn.get("1.field.kernel_raw", 0.0)
                        print(f"  kernel: mean={kernel.mean().item():.4f} "
                              f"min={kernel.min().item():.4f} max={kernel.max().item():.4f} "
                              f"grad={kg:.6f}")
                    elif field.column_attr is not None:
                        print(f"  column_attr: mean={field.column_attr.mean().item():.6f} "
                              f"std={field.column_attr.std().item():.6f}")
                    elif field.full_raw is not None:
                        print(f"  full_attention: std={field.full_raw.std().item():.6f}")
                    elif hasattr(field, "energy_score_gain"):
                        print(f"  energy_score: gain={field.energy_score_gain.item():.4f} "
                              f"sign={field.energy_score_sign.item():.4f}")
                    elif field.full_raw is not None:
                        print(f"  full_attention: std={field.full_raw.std().item():.6f}")
                field.disable_stats()

        if args.model == "rect_nf" and args.rect_curriculum and field.active_W < field.W:
            if grow_ready:
                field.set_active_W(field.active_W + 1)
                print(f"  curriculum: reached right edge, active_W={field.active_W}/{field.W}")
            else:
                print(f"  curriculum: waiting at active_W={field.active_W}/{field.W}")

        acc = evaluate(model, test_loader, device)
        el = time.time() - t0
        print(f"[{args.model}] ep{ep+1}/{args.epochs} "
              f"loss={run_loss/run_iters:.4f} test_acc={acc:.4f} elapsed={el:.0f}s")

    print(f"FINAL {args.model}: test_acc={acc:.4f}")


if __name__ == "__main__":
    main()
