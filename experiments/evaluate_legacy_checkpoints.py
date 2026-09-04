"""Evaluate the pre-existing original_model.pt / retrained_model.pt checkpoints
and import them as the official Phase 03 (M_old) / Phase 04 (M_retrain) results.

These two checkpoints were trained by the original main.py / retrain_baseline.py
scripts (before the repo refactor), not by experiments/run_cifar100_fl.py or
experiments/run_full_retraining.py. This script does NOT retrain anything —
it only runs evaluation, so the "no fabricated results" rule still holds:
every number here comes from an actual forward pass over real data.

Caveat (important, written to the output files too):
  - original_model.pt (M_old) was trained with an UNSEEDED partition, so we
    cannot know which samples belonged to which client. Only overall test
    accuracy is reported for it.
  - retrained_model.pt (M_retrain) WAS trained with set_seed(42) before
    partitioning, in retrain_baseline.py, using numpy's *global* RNG
    (np.random.shuffle / np.random.dirichlet). To identify its forget-client
    (client 0) samples, this script reproduces that exact legacy partition
    function verbatim — the new src/data/partition.py uses a different RNG
    API (np.random.default_rng) and will NOT reproduce the same split.

Usage:
    python experiments/evaluate_legacy_checkpoints.py \
        --original path/to/original_model.pt \
        --retrained path/to/retrained_model.pt \
        --data-dir data/raw
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.cifar import get_cifar100_datasets
from src.models.resnet import build_resnet18


# --- Verbatim copy of retrain_baseline.py's partitioning, for reproducing ---
# --- the exact forget-client split it used (do not "improve" this; any    ---
# --- change would break the reproduction).                                ---
def _legacy_set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _legacy_partition_dirichlet(dataset, num_clients=5, alpha=0.5):
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        np.random.shuffle(idx_c)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.array(
            [p * (len(idx_j) < len(dataset) / num_clients) for p, idx_j in zip(proportions, client_indices)]
        )
        if proportions.sum() == 0:
            proportions = np.ones(num_clients) / num_clients
        else:
            proportions = proportions / proportions.sum()

        split_sizes = (proportions * len(idx_c)).astype(int)
        split_sizes[-1] = len(idx_c) - split_sizes[:-1].sum()

        splits = np.split(idx_c, np.cumsum(split_sizes)[:-1])
        for client_id in range(num_clients):
            client_indices[client_id].extend(splits[client_id].tolist())

    return client_indices
# --- end verbatim copy ---


@torch.no_grad()
def evaluate(model, dataset, device, batch_size=128):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def main(original_ckpt: str, retrained_ckpt: str, data_dir: str, device: str) -> None:
    device = torch.device(device)
    _, test_set = get_cifar100_datasets(data_dir)
    train_set, _ = get_cifar100_datasets(data_dir)  # needed only for legacy partition reconstruction

    # ---- M_old (original_model.pt) ----
    m_old_dir = Path("artifacts/experiments/cifar100_fl")
    m_old_dir.mkdir(parents=True, exist_ok=True)
    m_old = build_resnet18(num_classes=100).to(device)
    m_old.load_state_dict(torch.load(original_ckpt, map_location=device))
    m_old_test_acc = evaluate(m_old, test_set, device)

    with (m_old_dir / "imported_metrics.json").open("w") as f:
        json.dump(
            {
                "source": "imported from pre-refactor main.py checkpoint (original_model.pt)",
                "overall_test_accuracy": m_old_test_acc,
                "note": "Per-client / forget-client accuracy not available: "
                "main.py's partition was unseeded, so client assignment cannot be reconstructed.",
            },
            f,
            indent=2,
        )
    torch.save(m_old.state_dict(), m_old_dir / "imported_original_model.pt")
    print(f"M_old overall test accuracy: {m_old_test_acc:.4f}")

    # ---- M_retrain (retrained_model.pt) ----
    m_retrain_dir = Path("artifacts/experiments/full_retraining")
    m_retrain_dir.mkdir(parents=True, exist_ok=True)

    _legacy_set_seed(42)
    client_indices = _legacy_partition_dirichlet(train_set, num_clients=5, alpha=0.5)
    forget_indices = client_indices[0]
    forget_set = Subset(train_set, forget_indices)

    m_retrain = build_resnet18(num_classes=100).to(device)
    m_retrain.load_state_dict(torch.load(retrained_ckpt, map_location=device))
    m_retrain_test_acc = evaluate(m_retrain, test_set, device)
    m_retrain_forget_acc = evaluate(m_retrain, forget_set, device)

    with (m_retrain_dir / "imported_metrics.json").open("w") as f:
        json.dump(
            {
                "source": "imported from pre-refactor retrain_baseline.py checkpoint (retrained_model.pt)",
                "forget_client_id": 0,
                "num_forget_samples": len(forget_indices),
                "overall_test_accuracy": m_retrain_test_acc,
                "forget_client_accuracy": m_retrain_forget_acc,
                "note": "Forget-client split reconstructed by re-running retrain_baseline.py's "
                "exact legacy partition_dirichlet() with set_seed(42), since the new "
                "src/data/partition.py uses a different RNG API and would not reproduce it.",
            },
            f,
            indent=2,
        )
    torch.save(m_retrain.state_dict(), m_retrain_dir / "imported_retrained_model.pt")
    print(f"M_retrain overall test accuracy: {m_retrain_test_acc:.4f}")
    print(f"M_retrain forget-client accuracy: {m_retrain_forget_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--retrained", required=True)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    main(args.original, args.retrained, args.data_dir, args.device)
