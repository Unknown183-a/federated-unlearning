"""Run Phase 05: Gradient Ascent unlearning.

Loads M_old, runs gradient ascent on the forget client's data only,
and evaluates the result (M_unlearn) on both the held-out test set and
the forget client's own data — before and after — so the accuracy/
forgetting tradeoff is visible directly, rather than needing a
separate evaluation pass.

Usage:
    python experiments/run_gradient_ascent.py \
        --base-config configs/cifar100_fl.yaml \
        --unlearning-config configs/unlearning.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from src.data.cifar import get_cifar100_datasets
from src.data.partition import partition_dataset
from src.federated.client import FederatedClient
from src.models.resnet import build_resnet18
from src.unlearning.gradient_ascent import GradientAscentUnlearner
from src.utils.config import load_config, save_config


def main(base_config_path: str, unlearning_config_path: str) -> None:
    base_cfg = load_config(base_config_path)
    unlearn_cfg = load_config(unlearning_config_path)
    torch.manual_seed(unlearn_cfg.seed)

    artifacts_dir = Path(unlearn_cfg.output.artifacts_dir) / "gradient_ascent"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    save_config(
        {"base_config": base_config_path, "unlearning_config": unlearning_config_path},
        artifacts_dir / "config.yaml",
    )

    device = base_cfg.device
    train_set, test_set = get_cifar100_datasets(base_cfg.data.data_dir)
    client_datasets = partition_dataset(
        train_set,
        num_clients=base_cfg.data.num_clients,
        strategy=base_cfg.data.partition,
        alpha=base_cfg.data.get("non_iid_alpha", 0.5),
        seed=base_cfg.seed,
    )
    forget_client_id = unlearn_cfg.forget_client_id
    forget_dataset = client_datasets[forget_client_id]

    ga_cfg = unlearn_cfg.gradient_ascent
    forget_loader = DataLoader(forget_dataset, batch_size=ga_cfg.batch_size, shuffle=True)

    model = build_resnet18(num_classes=base_cfg.model.num_classes).to(device)
    model.load_state_dict(torch.load(unlearn_cfg.base_checkpoint, map_location=device))

    test_client = FederatedClient(-1, test_set, batch_size=base_cfg.data.batch_size, device=device)
    forget_client = FederatedClient(
        forget_client_id, forget_dataset, batch_size=base_cfg.data.batch_size, device=device
    )

    before = {"test": test_client.evaluate(model), "forget": forget_client.evaluate(model)}
    print("Before unlearning:", before)

    unlearner = GradientAscentUnlearner(model, config={**ga_cfg, "device": device})
    m_unlearn = unlearner.unlearn(forget_loader)

    after = {"test": test_client.evaluate(m_unlearn), "forget": forget_client.evaluate(m_unlearn)}
    print("After unlearning: ", after)

    torch.save(m_unlearn.state_dict(), artifacts_dir / "m_unlearn_final.pt")
    with (artifacts_dir / "metrics.json").open("w") as f:
        json.dump({"before": before, "after": after}, f, indent=2)

    print(f"Done. M_unlearn artifacts written to {artifacts_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/cifar100_fl.yaml")
    parser.add_argument("--unlearning-config", default="configs/unlearning.yaml")
    args = parser.parse_args()
    main(args.base_config, args.unlearning_config)
