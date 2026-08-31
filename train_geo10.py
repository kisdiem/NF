"""MNIST probe: ten fixed geometric class points for the rectangular NF."""
import argparse
import time

import torch
import torch.nn.functional as F

from nf_field import Geo10RectNFBlock
from train_mnist import load_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--subset", type=int, default=5000)
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--route", default="all",
                    choices=["all", "diagonal", "parallel_ru", "parallel_rud",
                             "receiver_mix", "inbound_kernel", "inbound_5",
                             "inbound_full"])
    ap.add_argument("--residual", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", default="data/mnist")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_loader, test_loader = load_data(args.data_root, args.batch, args.subset)
    cfg = dict(W=args.width, tau_a=0.2, tau_p=1.0,
               residual_alpha=args.residual, gain_init=1.0,
               train_gain=True, energy_mode="linear", energy_scale=1.0,
               route_mode=args.route, threshold_init=0.5)
    model = Geo10RectNFBlock(784, args.height, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"[geo10] device={device} epochs={args.epochs} lr={args.lr} "
          f"subset={args.subset} H={args.height} W={args.width} "
          f"route={args.route} residual={args.residual}")

    def evaluate():
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                logits = model(x.to(device).flatten(1))
                correct += (logits.argmax(1) == y.to(device)).sum().item()
                total += y.numel()
        return correct / total

    start = time.time()
    for ep in range(args.epochs):
        model.train()
        total_loss = total = 0
        for x, y in train_loader:
            x, y = x.to(device).flatten(1), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * y.numel()
            total += y.numel()
        acc = evaluate()
        print(f"[geo10] ep{ep + 1}/{args.epochs} "
              f"loss={total_loss / total:.4f} test_acc={acc:.4f} "
              f"elapsed={time.time() - start:.0f}s")
    print(f"FINAL geo10: test_acc={acc:.4f}")


if __name__ == "__main__":
    main()
