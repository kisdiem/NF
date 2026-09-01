"""Common interface for Phase-1 intrinsic-training strategies."""
from abc import ABC, abstractmethod

import torch


class TrainingStrategy(ABC):
    def __init__(self, model, inventory, lr, weight_decay=1e-4):
        self.model = model
        self.inventory = inventory
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self._original_requires_grad = {
            name: p.requires_grad for name, p in model.named_parameters()
        }

    def prepare_batch(self, batch_index):
        """Called before forward so E2 can remove inactive leaves from graph."""

    @abstractmethod
    def update(self, loss, batch_index):
        raise NotImplementedError

    def finish(self):
        for name, p in self.model.named_parameters():
            p.requires_grad_(self._original_requires_grad[name])

    @staticmethod
    def _backward_step(loss, optimizer, parameters, clip_norm):
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, clip_norm)
        optimizer.step()
        return float(grad_norm)
