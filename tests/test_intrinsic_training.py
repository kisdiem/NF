import torch
import torch.nn.functional as F

from experiments.intrinsic_training.run_phase1 import build_model
from training_strategies import AlternatingBP, classify_parameters


MODEL_KEYS = [
    "minimal_local_nf", "local_electrical_v1", "local_electrical_v2",
    "local_electrical_v3", "dynamic_nf", "hierarchical_nf",
    "bio_neuron", "directional_rect_v4", "discrete_nf_v3",
]


def test_registry_is_complete_and_disjoint():
    for key in MODEL_KEYS:
        model = build_model(key)
        inventory = classify_parameters(model, key)
        inventory.validate(model)
        assert inventory.intrinsic_names, key


def test_all_models_forward_and_constraints_are_finite():
    x = torch.rand(2, 1, 28, 28)
    for key in MODEL_KEYS:
        model = build_model(key)
        logits = model(x)
        assert logits.shape == (2, 10), key
        assert torch.isfinite(logits).all(), key


def test_alternating_optimizer_really_freezes_inactive_group():
    torch.manual_seed(0)
    model = build_model("minimal_local_nf")
    inventory = classify_parameters(model, "minimal_local_nf")
    strategy = AlternatingBP(model, inventory, 1e-3, synaptic_steps=1,
                             intrinsic_steps=1, weight_decay=1e-2,
                             optimizer_cls=torch.optim.AdamW)
    x, y = torch.rand(8, 1, 28, 28), torch.randint(0, 10, (8,))
    named = dict(model.named_parameters())

    intrinsic_before = {n: named[n].detach().clone()
                        for n in inventory.intrinsic_names}
    strategy.prepare_batch(0)
    strategy.update(F.cross_entropy(model(x), y), 0)
    assert all(torch.equal(named[n], old) for n, old in intrinsic_before.items())

    synaptic_before = {n: named[n].detach().clone()
                       for n in inventory.synaptic_names + inventory.other_names}
    strategy.prepare_batch(1)
    strategy.update(F.cross_entropy(model(x), y), 1)
    assert all(torch.equal(named[n], old) for n, old in synaptic_before.items())
    assert strategy.max_inactive_change == 0.0
    strategy.finish()
