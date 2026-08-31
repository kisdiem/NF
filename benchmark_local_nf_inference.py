"""Inference-only benchmark for the unfused and fused local convolutions."""
import argparse, json, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_nf import DynamicNFMLP
from local_electrical_nf_v3 import LocalElectricalNFV3MLP


class MLP(nn.Module):
    def __init__(self, activation):
        super().__init__(); self.fc1=nn.Linear(784,64); self.fc2=nn.Linear(64,10); self.activation=activation
    def forward(self, x): return self.fc2(self.activation(self.fc1(x)))


def measure(model, x, warmup, repeats):
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup): model(x)
        start = time.perf_counter()
        for _ in range(repeats): model(x)
        elapsed = time.perf_counter() - start
    return {"total_seconds": elapsed, "per_inference_ms": elapsed / repeats * 1000,
            "per_sample_ms": elapsed / repeats / x.shape[0] * 1000}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--device", default="cpu"); p.add_argument("--threads", type=int, default=1); p.add_argument("--warmup", type=int, default=10); p.add_argument("--repeats", type=int, default=100); args = p.parse_args()
    torch.set_num_threads(args.threads); device = torch.device(args.device)
    results = {}
    for batch in (1, 128):
        torch.manual_seed(0); normal = LocalElectricalNFV3MLP(8, 8, 4, mode="raw_bounded", fuse_local_convs=False, collect_diagnostics=False).to(device)
        fused = LocalElectricalNFV3MLP(8, 8, 4, mode="raw_bounded", fuse_local_convs=True, collect_diagnostics=False).to(device); fused.load_state_dict(normal.state_dict())
        x = torch.randn(batch, 784, device=device)
        with torch.inference_mode(): diff = float((normal(x) - fused(x)).abs().max())
        torch.manual_seed(0); relu = MLP(F.relu).to(device); gelu = MLP(F.gelu).to(device)
        torch.manual_seed(0); dynamic = DynamicNFMLP(hidden=64, n_nodes=16, node_dim=4, branches=4, steps=4, relation_gain_init=.1, temperature=1., norm=True).to(device)
        results[str(batch)] = {"relu_mlp": measure(relu, x, args.warmup, args.repeats), "gelu_mlp": measure(gelu, x, args.warmup, args.repeats), "dynamic_nf": measure(dynamic, x, args.warmup, args.repeats), "local_normal": measure(normal, x, args.warmup, args.repeats), "local_fused": measure(fused, x, args.warmup, args.repeats), "max_output_diff": diff}
    for batch, r in results.items():
        old, new = r["local_normal"]["per_inference_ms"], r["local_fused"]["per_inference_ms"]
        r["fuse_speedup_percent"] = (old - new) / old * 100
        print(f"batch={batch} relu={r['relu_mlp']['per_inference_ms']:.4f}ms gelu={r['gelu_mlp']['per_inference_ms']:.4f}ms dynamic={r['dynamic_nf']['per_inference_ms']:.4f}ms local={new:.4f}ms fused_speedup={r['fuse_speedup_percent']:.2f}% diff={r['max_output_diff']}")
    out = "local_electrical_nf_inference_results"; os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "cpu_threads1.json"), "w", encoding="utf-8") as f: json.dump(results, f, indent=2)


if __name__ == "__main__": main()
