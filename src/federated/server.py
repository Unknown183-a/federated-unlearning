"""Federated server: orchestrates rounds, selects clients, aggregates, checkpoints."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.logging import get_logger

from .client import FederatedClient
from .fedavg import fedavg
from .round_manager import RoundManager


class FederatedServer:
    def __init__(
        self,
        global_model: nn.Module,
        clients: Dict[int, FederatedClient],
        artifacts_dir: str | Path,
        save_every_n_rounds: int = 5,
    ):
        self.global_model = global_model
        self.clients = clients
        self.artifacts_dir = Path(artifacts_dir)
        self.save_every_n_rounds = save_every_n_rounds
        self.round_manager = RoundManager()
        self.logger = get_logger("federated.server", self.artifacts_dir / "training.log")

    def _resume_state(self) -> int:
        """Load the latest checkpoint (if any) into global_model and restore
        prior metrics history, so a rerun continues instead of restarting.

        Returns the round number to start from (1 if there's nothing to resume).
        """
        ckpt_dir = self.artifacts_dir / "checkpoints"
        checkpoints = sorted(ckpt_dir.glob("round_*.pt")) if ckpt_dir.exists() else []
        if not checkpoints:
            return 1

        latest = checkpoints[-1]
        payload = load_checkpoint(self.global_model, latest, map_location="cpu")
        last_round = payload.get("round", 0)

        metrics_path = self.artifacts_dir / "metrics.csv"
        if metrics_path.exists():
            with metrics_path.open(newline="") as f:
                for row in csv.DictReader(f):
                    parsed = {}
                    for k, v in row.items():
                        if k == "round":
                            parsed[k] = int(v)
                        else:
                            try:
                                parsed[k] = float(v)
                            except (TypeError, ValueError):
                                parsed[k] = v
                    self.round_manager.history.append(parsed)

        self.logger.info("Resuming from %s (completed through round %d)", latest.name, last_round)
        return last_round + 1

    def run(
        self,
        num_rounds: int,
        local_epochs: int,
        local_lr: float,
        test_client: FederatedClient | None = None,
        mu: float = 0.0,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        resume: bool = False,
    ):
        start_round = self._resume_state() if resume else 1
        if start_round > num_rounds:
            self.logger.info("Already completed %d/%d rounds; nothing to do.", start_round - 1, num_rounds)
            return self.global_model

        for round_num in range(start_round, num_rounds + 1):
            updates = [
                client.local_train(
                    self.global_model,
                    epochs=local_epochs,
                    lr=local_lr,
                    mu=mu,
                    momentum=momentum,
                    weight_decay=weight_decay,
                )
                for client in self.clients.values()
            ]
            new_state = fedavg(updates)
            self.global_model.load_state_dict(new_state)

            avg_train_loss = sum(u.train_loss for u in updates) / len(updates)
            metrics = {"avg_train_loss": avg_train_loss}
            if test_client is not None:
                metrics.update({f"test_{k}": v for k, v in test_client.evaluate(self.global_model).items()})

            self.round_manager.record(round_num, **metrics)
            self.logger.info("Round %d/%d: %s", round_num, num_rounds, metrics)

            if round_num % self.save_every_n_rounds == 0 or round_num == num_rounds:
                ckpt_path = self.artifacts_dir / "checkpoints" / f"round_{round_num:03d}.pt"
                save_checkpoint(self.global_model, ckpt_path, extra={"round": round_num, "metrics": metrics})
                # Save metrics.csv on every checkpoint too, not just at the end,
                # so a mid-run disconnect doesn't lose the history needed to resume.
                self.round_manager.save_csv(self.artifacts_dir / "metrics.csv")

        self.round_manager.save_csv(self.artifacts_dir / "metrics.csv")
        return self.global_model
