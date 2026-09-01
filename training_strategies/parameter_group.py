import torch

from .base import TrainingStrategy


class ParameterGroupBP(TrainingStrategy):
    """E1: task BP with a relative learning rate for intrinsic properties."""
    def __init__(self, model, inventory, lr, intrinsic_ratio=0.1,
                 weight_decay=1e-4, clip_norm=5.0,
                 optimizer_cls=torch.optim.Adam):
        super().__init__(model, inventory, lr, weight_decay)
        syn_names = set(inventory.synaptic_names + inventory.other_names)
        int_names = set(inventory.intrinsic_names)
        syn = [p for n, p in model.named_parameters() if n in syn_names and p.requires_grad]
        intrinsic = [p for n, p in model.named_parameters() if n in int_names and p.requires_grad]
        if not intrinsic:
            raise ValueError(f"{inventory.model_key} has no trainable intrinsic parameters")
        self.parameters = syn + intrinsic
        self.optimizer = optimizer_cls([
            {"params": syn, "lr": lr, "name": "synaptic"},
            {"params": intrinsic, "lr": lr * intrinsic_ratio, "name": "intrinsic"},
        ], weight_decay=weight_decay)
        self.intrinsic_ratio = float(intrinsic_ratio)
        self.clip_norm = clip_norm

    def update(self, loss, batch_index):
        return self._backward_step(loss, self.optimizer, self.parameters, self.clip_norm)
