"""Teacher-function probe for the NF nonlinearity.

This deliberately removes MNIST, W_up/W_down and the residual bypass.  The
only question is whether the field itself can learn a useful nonlinear map:

    h -> ReLU(h)

Run from the repository root, for example:
    py -3 teacher_nf.py --steps 1200
"""
import argparse
import math
import random

import torch
import torch.nn.functional as F

from nf_field import DirectionalRectNeuralField


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_linear(x, y):
    """Best affine map y ~= xW+b, included as a useful reference."""
    xa = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device)], dim=1)
    sol = torch.linalg.lstsq(xa, y).solution
    pred = xa @ sol
    return F.mse_loss(pred, y).item()


def run_one(mode, width, x, y, steps, lr, threshold, tau_a, device):
    field = DirectionalRectNeuralField(
        d=x.shape[1], W=width, tau_a=tau_a, tau_p=1.0,
        residual_alpha=0.0, gain_init=1.0, train_gain=True,
        energy_mode="linear", route_mode=mode, threshold_init=threshold,
    ).to(device)
    opt = torch.optim.Adam(field.parameters(), lr=lr)
    best = math.inf
    best_step = 0
    with torch.no_grad():
        initial = F.mse_loss(field(x), y).item()
    for step in range(1, steps + 1):
        # Fixed teacher set: this tests memorisation of a random nonlinear map
        # only through the field's shared parameters, not data augmentation.
        pred = field(x)
        loss = F.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(field.parameters(), 10.0)
        opt.step()
        value = float(loss.detach())
        if value < best:
            best, best_step = value, step

    with torch.no_grad():
        pred = field(x)
        mse = F.mse_loss(pred, y).item()
        corr = torch.corrcoef(torch.stack([pred.flatten(), y.flatten()]))[0, 1].item()
        sign_acc = ((pred >= 0) == (y >= 0)).float().mean().item()
        first_energy = float(field.relay_energy_first)
        last_energy = float(field.relay_energy_last)
        ratio = float(field.relay_ratio)
    return {
        "mode": mode, "W": width, "initial": initial, "best": best,
        "final": mse, "best_step": best_step, "corr": corr,
        "sign_acc": sign_acc, "first": first_energy, "last": last_energy,
        "ratio": ratio,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--samples", type=int, default=2048)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tau-a", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    seed_all(args.seed)
    device = torch.device(args.device)
    x = torch.randn(args.samples, args.dim, device=device)
    y = F.relu(x)
    linear_mse = fit_linear(x, y)
    print(f"device={device} samples={args.samples} dim={args.dim} steps={args.steps}")
    print(f"linear_affine_mse={linear_mse:.6f}")
    print("mode\tW\tinitial\tbest\tfinal\tcorr\tsign_acc\tfirst->last\tratio")
    # inbound_kernel is the cleanest receiver-centric version; all is the
    # current original three-direction field; inbound_full is the flexible
    # upper-bound-like version tested previously.
    for width in (1, 16):
        for mode in ("all", "inbound_kernel", "inbound_full"):
            result = run_one(mode, width, x, y, args.steps, args.lr,
                             args.threshold, args.tau_a, device)
            print(
                f"{result['mode']}\t{result['W']}\t"
                f"{result['initial']:.6f}\t{result['best']:.6f}\t"
                f"{result['final']:.6f}\t{result['corr']:.4f}\t"
                f"{result['sign_acc']:.4f}\t"
                f"{result['first']:.4f}->{result['last']:.4f}\t"
                f"{result['ratio']:.4f}"
            )


if __name__ == "__main__":
    main()
