import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from train_mnist import build_model, load_data, evaluate


def run_variant(name, route_mode, out_dir):
    seed = 0
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = load_data("data/mnist", 128, 5000)
    model = build_model("rect_nf", {
        "W": 16, "height": 64, "route_mode": route_mode,
        "energy_mode": "linear", "residual_alpha": 1.0,
    }).to(device)
    # 90.4 version: fixed overall input energy, no relay objective.
    model[1].input_scale = 3.0
    opt = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    field = model[1].field
    trace = []

    for epoch in range(10):
        model.train()
        for batch_index, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            field.enable_stats()
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if batch_index == 0:
                stats = field._stats
                energies = [float(v) for v in stats["e_abs_means"]]
                activities = [float(v) for v in stats["z_means"]]
                trace.append({
                    "epoch": epoch + 1,
                    "loss_first_batch": float(loss.item()),
                    "energy_by_column": energies,
                    "activity_by_column": activities,
                    "first_energy": energies[0],
                    "last_energy": energies[-1],
                    "last_to_first_ratio": energies[-1] / (energies[0] + 1e-8),
                })
                field.disable_stats()
        acc = evaluate(model, test_loader, device)
        trace[-1]["test_accuracy"] = float(acc)

    payload = {
        "name": name,
        "description": "Rectangular neural field energy trace; first training batch of each epoch",
        "parameters": {
            "seed": seed, "device": device, "subset": 5000,
            "batch": 128, "epochs": 10, "learning_rate": 0.003,
            "height": 64, "width": 16, "input_scale": 3.0,
            "tau_a": 0.2, "tau_p": 1.0, "residual_alpha": 1.0,
            "gain_init": 1.0, "train_gain": True,
            "energy_mode": "linear", "route_mode": route_mode,
            "relay_loss_weight": 0.0,
        },
        "trace": trace,
    }
    json_path = out_dir / f"{name}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def plot_payloads(payloads, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for payload in payloads:
        label = payload["name"]
        for item in payload["trace"]:
            axes[0].plot(range(1, 17), item["energy_by_column"], alpha=0.25, color=None)
        final = payload["trace"][-1]["energy_by_column"]
        axes[0].plot(range(1, 17), final, linewidth=2.5, label=label)
        axes[1].plot(
            [item["epoch"] for item in payload["trace"]],
            [item["last_to_first_ratio"] for item in payload["trace"]],
            marker="o", linewidth=2, label=label,
        )
    axes[0].set_title("Energy by column (final epoch, plus epoch traces)")
    axes[0].set_xlabel("Field column")
    axes[0].set_ylabel("Mean absolute energy")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].set_title("Last-column / first-column energy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Energy ratio")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.savefig(out_dir / "energy_comparison.png", dpi=180)
    plt.close(fig)


def main():
    out_dir = Path("energy_report")
    out_dir.mkdir(exist_ok=True)
    payloads = [
        run_variant("best_90p4_all_routes", "all", out_dir),
        run_variant("diagonal_up_right_down_right", "diagonal", out_dir),
    ]
    plot_payloads(payloads, out_dir)
    (out_dir / "README.txt").write_text(
        "Two configurations and traces are included.\n"
        "Each trace records the first training batch of every epoch.\n"
        "energy_by_column uses the 16 field columns, before the final residual readout.\n",
        encoding="utf-8",
    )
    print(f"saved {out_dir.resolve()}")
    for payload in payloads:
        last = payload["trace"][-1]
        best = max(item["test_accuracy"] for item in payload["trace"])
        print(payload["name"], "best_acc=", round(best, 4),
              "final_ratio=", round(last["last_to_first_ratio"], 4))


if __name__ == "__main__":
    main()
