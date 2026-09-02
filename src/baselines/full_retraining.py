"""Full-retraining baseline: gold-standard reference for unlearning quality.

Reuses `FederatedServer` (src/federated/server.py) on the client set with
the forget client excluded, so M_retrain is trained under exactly the
same loop/aggregation used to produce M_old — the only difference is
which clients participate.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict

import torch.nn as nn
from torch.utils.data import Dataset

from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.utils.checkpoint import save_checkpoint


def run_full_retraining(
    client_datasets: Dict[int, Dataset],
    forget_client_id: int,
    model_factory: Callable[[], nn.Module],
    config: Dict[str, Any],
) -> nn.Module:
    """Train M_retrain from scratch on every client except `forget_client_id`.

    config keys:
        num_rounds, local_epochs, local_lr — from configs/unlearning.yaml's
            `full_retraining` section (never hard-coded, per repo convention).
        batch_size, device                 — from the base FL config, so
            M_retrain trains under the same conditions as M_old.
        artifacts_dir                      — where checkpoints/metrics.csv/
            the final M_retrain checkpoint are written.
        save_every_n_rounds                — passed straight to FederatedServer.
        test_dataset (optional)            — held-out test set for per-round
            accuracy tracking during retraining.
    """
    device = config.get("device", "cpu")
    batch_size = config.get("batch_size", 32)
    artifacts_dir = Path(config["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    retained_clients = {
        cid: FederatedClient(cid, ds, batch_size=batch_size, device=device)
        for cid, ds in client_datasets.items()
        if cid != forget_client_id
    }
    if not retained_clients:
        raise ValueError("No clients left to retrain on after excluding forget_client_id.")

    test_client = None
    if config.get("test_dataset") is not None:
        test_client = FederatedClient(-1, config["test_dataset"], batch_size=batch_size, device=device)

    server = FederatedServer(
        global_model=model_factory(),
        clients=retained_clients,
        artifacts_dir=artifacts_dir,
        save_every_n_rounds=config.get("save_every_n_rounds", 5),
    )
    m_retrain = server.run(
        num_rounds=config["num_rounds"],
        local_epochs=config["local_epochs"],
        local_lr=config["local_lr"],
        test_client=test_client,
    )

    save_checkpoint(m_retrain, artifacts_dir / "m_retrain_final.pt", extra={"forget_client_id": forget_client_id})

    # Forget-client accuracy is evaluated but the client never trains on
    # M_retrain — this is the gold-standard "how much does it still know
    # about the excluded client" number that M_unlearn gets compared against.
    if forget_client_id in client_datasets:
        forget_client = FederatedClient(
            forget_client_id, client_datasets[forget_client_id], batch_size=batch_size, device=device
        )
        forget_metrics = forget_client.evaluate(m_retrain)
        with (artifacts_dir / "forget_client_metrics.json").open("w") as f:
            json.dump(forget_metrics, f, indent=2)

    return m_retrain
