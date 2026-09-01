import torch

from .base import TrainingStrategy


class JointBP(TrainingStrategy):
    """E0: original joint task-loss backpropagation."""
    def __init__(self, model, inventory, lr, weight_decay=1e-4, clip_norm=5.0,
                 optimizer_cls=torch.optim.Adam):
        super().__init__(model, inventory, lr, weight_decay)
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = optimizer_cls(self.parameters, lr=lr, weight_decay=weight_decay)
        self.clip_norm = clip_norm

    def update(self, loss, batch_index):
        return self._backward_step(loss, self.optimizer, self.parameters, self.clip_norm)
