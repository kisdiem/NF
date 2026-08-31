"""Smoke tests for the discrete neural field operator.

Run:  python tests/test_field.py

Note: torch.autograd.gradcheck is only valid on *smooth* sub-paths. The hard
threshold + STE makes the full propagation non-differentiable at thresholds, so
we gradcheck the bilinear write (fully smooth), hand-check two degenerate
forward cases, verify backward produces gradients everywhere, and check
reproducibility.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nf_field import DiscreteNeuralField, NFMLPBlock

SOFTPLUS_INV_1 = 0.5413248546122981  # s such that softplus(s) == 1


def field_small(**kw):
    cfg = dict(d=4, H=4, W=4, K_s=1, K_t=1, L_max=1, D_max=1, R=1)
    cfg.update(kw)
    return DiscreteNeuralField(**cfg)


def test_write_gradcheck():
    # gradcheck requires double precision (float32 finite-difference is too coarse)
    f = field_small(K_s=2).double()
    h = torch.randn(3, 4, dtype=torch.float64)
    fn = lambda *args: f._write(h).sum()
    ok = torch.autograd.gradcheck(fn, (f.A, f.c, f.w, f.b),
                                  eps=1e-6, atol=1e-6, rtol=1e-3)
    assert ok, "bilinear write failed gradient check"
    print("PASS _write gradcheck")


def test_no_activation_outputs_zero():
    f = field_small()
    with torch.no_grad():
        f.T.fill_(100.0)                       # nothing reaches threshold
        f.s_param.fill_(SOFTPLUS_INV_1)        # S = 1
    h = torch.randn(4, 4)
    v = f(h)
    assert torch.allclose(v, torch.zeros(4, 4), atol=1e-6), f"expected zeros, got {v}"
    print("PASS no-activation -> v=0")


def test_saturated_outputs_ones():
    f = field_small()
    with torch.no_grad():
        f.T.fill_(-100.0)                      # everything fires
        f.s_param.fill_(SOFTPLUS_INV_1)        # S = 1, D = 1 -> m = 1
    h = torch.randn(4, 4)
    v = f(h)
    assert torch.allclose(v, torch.ones(4, 4), atol=1e-5), f"expected ones, got {v}"
    print("PASS saturated -> v=1 (region pooling path)")


def test_backward_flows():
    f = NFMLPBlock(16, 4, 3, dict(H=4, W=4, K_s=2, K_t=4, L_max=2, D_max=2, R=3))
    x = torch.randn(5, 16)
    y = torch.randint(0, 3, (5,))
    loss = torch.nn.functional.cross_entropy(f(x), y)
    loss.backward()
    names = [n for n, p in f.named_parameters()]
    assert names, "no parameters"
    for n, p in f.named_parameters():
        assert p.grad is not None, f"param {n} got no gradient"
        assert p.grad.norm() > 0, f"param {n} got zero gradient"
    print(f"PASS backward flows through {len(names)} params (STE path alive)")


def test_reproducible():
    f = field_small(K_s=2, K_t=4, L_max=2, D_max=2, R=3)
    h = torch.randn(4, 4)
    v1, v2 = f(h), f(h)
    assert torch.equal(v1, v2), "forward is not deterministic"
    print("PASS reproducible")


def test_not_always_dead():
    # default init should produce some activity for typical inputs
    f = DiscreteNeuralField(d=64, H=32, W=32, K_s=16, K_t=8, L_max=4, D_max=3, R=4)
    any_act = False
    with torch.no_grad():
        for _ in range(10):
            v = f(torch.randn(8, 64))
            if v.sum() > 0:
                any_act = True
                break
    print("not_always_dead:", any_act)
    assert any_act, "field produces zero activity on all random inputs"


def test_forward_seq():
    # sequence feeding: shape check + backward flows
    f = DiscreteNeuralField(d=4, H=4, W=4, K_s=2, K_t=4, L_max=2, D_max=2, R=3)
    h_seq = torch.randn(3, 4, 4)
    v = f.forward_seq(h_seq)
    assert v.shape == (3, 4)
    v.sum().backward()
    for n, p in f.named_parameters():
        assert p.grad is not None and p.grad.norm() > 0, f"param {n} got no gradient"
    print("PASS forward_seq shape + backward")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_write_gradcheck()
    test_no_activation_outputs_zero()
    test_saturated_outputs_ones()
    test_backward_flows()
    test_reproducible()
    test_not_always_dead()
    test_forward_seq()
    print("\nALL TESTS PASSED")
