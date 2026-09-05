"""Gradient Ascent unlearner.

Trains the model to *increase* its loss on the forget client's data —
the opposite of normal training — by minimizing the negative
cross-entropy (`gradient_ascent_loss`). This directly damages whatever
the model learned from the forget client, at the risk of also damaging
unrelated performance if run too long (see Phase 08's evaluation for
how that tradeoff gets measured).
"""
from __future__ import annotations

import copy
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .losses import gradient_ascent_loss


class GradientAscentUnlearner:
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = copy.deepcopy(model)
        self.config = config

    def unlearn(self, forget_loader: DataLoader) -> nn.Module:
        """Run gradient ascent on `forget_loader` for `config['epochs']` epochs.

        config keys: lr, epochs (required, from configs/unlearning.yaml's
        `gradient_ascent` section); device (optional, defaults to "cpu");
        max_grad_norm (optional, defaults to 1.0). Gradient clipping matters
        here specifically because minimizing negative cross-entropy is an
        unbounded objective — without clipping, weights can diverge to NaN
        within a handful of steps.
        """
        device = self.config.get("device", "cpu")
        self.model.to(device)
        self.model.train()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config["lr"])
        max_grad_norm = self.config.get("max_grad_norm", 1.0)

        for _ in range(self.config["epochs"]):
            for x, y in forget_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = gradient_ascent_loss(self.model(x), y)
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()

        return self.model
