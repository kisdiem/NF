import torch

from .base import TrainingStrategy


class AlternatingBP(TrainingStrategy):
    """E2: separate optimizers and genuinely frozen inactive parameters."""
    def __init__(self, model, inventory, lr, synaptic_steps=5, intrinsic_steps=1,
                 weight_decay=1e-4, clip_norm=5.0,
                 optimizer_cls=torch.optim.Adam):
        super().__init__(model, inventory, lr, weight_decay)
        syn_names = set(inventory.synaptic_names + inventory.other_names)
        int_names = set(inventory.intrinsic_names)
        named = dict(model.named_parameters())
        self.synaptic = [named[n] for n in syn_names if named[n].requires_grad]
        self.intrinsic = [named[n] for n in int_names if named[n].requires_grad]
        if not self.intrinsic:
            raise ValueError(f"{inventory.model_key} has no trainable intrinsic parameters")
        self.syn_names, self.int_names = syn_names, int_names
        self.syn_optimizer = optimizer_cls(self.synaptic, lr=lr, weight_decay=weight_decay)
        self.int_optimizer = optimizer_cls(self.intrinsic, lr=lr, weight_decay=weight_decay)
        self.synaptic_steps = int(synaptic_steps)
        self.intrinsic_steps = int(intrinsic_steps)
        self.cycle = self.synaptic_steps + self.intrinsic_steps
        self.clip_norm = clip_norm
        self.active_group = None
        self.max_inactive_change = 0.0

    def prepare_batch(self, batch_index):
        self.active_group = ("synaptic" if batch_index % self.cycle < self.synaptic_steps
                             else "intrinsic")
        for name, p in self.model.named_parameters():
            if name in self.syn_names:
                p.requires_grad_(self.active_group == "synaptic")
            elif name in self.int_names:
                p.requires_grad_(self.active_group == "intrinsic")

    def update(self, loss, batch_index):
        if self.active_group == "synaptic":
            inactive = self.intrinsic
            before = [p.detach().clone() for p in inactive]
            value = self._backward_step(loss, self.syn_optimizer, self.synaptic, self.clip_norm)
        else:
            inactive = self.synaptic
            before = [p.detach().clone() for p in inactive]
            value = self._backward_step(loss, self.int_optimizer, self.intrinsic, self.clip_norm)
        if before:
            change = max(float((p.detach() - old).abs().max())
                         for p, old in zip(inactive, before))
            self.max_inactive_change = max(self.max_inactive_change, change)
            if change != 0.0:
                raise RuntimeError(f"E2 inactive parameter changed by {change}")
        return value
