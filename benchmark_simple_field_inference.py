"""Inference-only timing for the simple wide-field MNIST models."""
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from train_simple_field_ab import FieldMLP


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = datasets.MNIST("data/mnist", train=False, download=True, transform=transforms.ToTensor())
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    x_fixed = next(iter(loader))[0].to(device)
    models = {
        "relu": nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)),
        "gelu": nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.GELU(), nn.Linear(256, 10)),
        "A_membrane_only": FieldMLP(False),
        "B_membrane_threshold": FieldMLP(True),
    }
    print(f"device={device} batch=128 test_samples={len(ds)}")
    for name, model in models.items():
        model.to(device).eval()
        with torch.inference_mode():
            for _ in range(20): model(x_fixed)
            if device.type == "cuda": torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(100): model(x_fixed)
            if device.type == "cuda": torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / 100
        print(f"{name}: {elapsed*1000:.4f}ms/batch, {elapsed/x_fixed.shape[0]*1000:.5f}ms/sample")


if __name__ == "__main__": main()
