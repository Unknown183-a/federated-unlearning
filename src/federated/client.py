"""Federated client: local training and evaluation on a private shard."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class ClientUpdate:
    client_id: int
    state_dict: Dict[str, torch.Tensor]
    num_samples: int
    train_loss: float


class FederatedClient:
    """Holds one client's private data and knows how to train/eval locally."""

    def __init__(self, client_id: int, dataset: Dataset, batch_size: int = 32, device: str = "cpu"):
        self.client_id = client_id
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device

    def local_train(
        self,
        global_model: nn.Module,
        epochs: int,
        lr: float,
        mu: float = 0.0,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> ClientUpdate:
        """Train a local copy of the global model for `epochs` epochs and return the update.

        `mu` is the FedProx proximal term weight (Li et al., 2018):
        adds (mu/2) * ||w - w_global||^2 to the loss, penalizing local
        drift away from the global model. `mu=0` recovers plain FedAvg
        local training exactly — this is the default so existing callers
        (e.g. the MNIST pipeline-validation experiment) are unaffected.

        FedProx's specific purpose is to make training stable with *more*
        local epochs on non-IID data (rather than needing to keep epochs
        low to avoid client drift, which is the plain-FedAvg workaround).
        """
        model = copy.deepcopy(global_model).to(self.device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        global_params = [p.detach().clone().to(self.device) for p in global_model.parameters()]

        total_loss, num_batches = 0.0, 0
        for _ in range(epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                if mu > 0:
                    prox_term = sum(
                        (w - w_glob).pow(2).sum() for w, w_glob in zip(model.parameters(), global_params)
                    )
                    loss = loss + (mu / 2) * prox_term
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return ClientUpdate(
            client_id=self.client_id,
            state_dict=model.state_dict(),
            num_samples=len(self.dataset),
            train_loss=avg_loss,
        )

    @torch.no_grad()
    def evaluate(self, model: nn.Module) -> Dict[str, float]:
        model = model.to(self.device)
        model.eval()
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        criterion = nn.CrossEntropyLoss()

        correct, total, total_loss = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)

        return {"accuracy": correct / max(total, 1), "loss": total_loss / max(total, 1)}
