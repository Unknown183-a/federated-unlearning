"""Run Phase 07: the joint Gradient Ascent + Knowledge Distillation engine.

This is the actual `M_unlearn` this thesis is about — unlike Phase 05
(GA alone) and Phase 06 (sequential GA->KD, shown to undo forgetting),
this jointly optimizes both objectives in one training loop per
docs/methodology.md's combined loss:

    L_total = lambda_forget * L_forget + lambda_kd * L_KD

Usage:
    python experiments/run_unlearning_engine.py \
        --base-config configs/cifar100_fl_colab.yaml \
        --unlearning-config configs/unlearning.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import ConcatDataset, DataLoader

from src.data.cifar import get_cifar100_datasets
from src.data.partition import partition_dataset
from src.federated.client import FederatedClient
from src.models.resnet import build_resnet18
from src.unlearning.engine import UnlearningEngine
from src.unlearning.forget_client import remaining_clients, select_forget_client
from src.utils.config import load_config, save_config


def main(base_config_path: str, unlearning_config_path: str) -> None:
    base_cfg = load_config(base_config_path)
    unlearn_cfg = load_config(unlearning_config_path)
    torch.manual_seed(unlearn_cfg.seed)

    artifacts_dir = Path(unlearn_cfg.output.artifacts_dir) / "engine"
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
    forget_dataset = select_forget_client(client_datasets, forget_client_id)
    remaining_dataset = ConcatDataset(list(remaining_clients(client_datasets, forget_client_id).values()))

    ga_cfg = unlearn_cfg.gradient_ascent
    forget_loader = DataLoader(forget_dataset, batch_size=ga_cfg.batch_size, shuffle=True)
    remaining_loader = DataLoader(remaining_dataset, batch_size=base_cfg.data.batch_size, shuffle=True)

    m_old = build_resnet18(num_classes=base_cfg.model.num_classes).to(device)
    m_old.load_state_dict(torch.load(unlearn_cfg.base_checkpoint, map_location=device))

    test_client = FederatedClient(-1, test_set, batch_size=base_cfg.data.batch_size, device=device)
    forget_client = FederatedClient(forget_client_id, forget_dataset, batch_size=base_cfg.data.batch_size, device=device)

    before = {"test": test_client.evaluate(m_old), "forget": forget_client.evaluate(m_old)}
    print("Before unlearning (M_old):", before)

    kd_cfg = unlearn_cfg.knowledge_distillation
    engine = UnlearningEngine(m_old, config={**kd_cfg, "max_grad_norm": ga_cfg.get("max_grad_norm", 1.0), "device": device})
    result = engine.run(forget_client_id=forget_client_id, forget_loader=forget_loader, remaining_loader=remaining_loader)
    m_unlearn = result.model

    after = {"test": test_client.evaluate(m_unlearn), "forget": forget_client.evaluate(m_unlearn)}
    print("After unlearning (M_unlearn):", after)

    torch.save(m_unlearn.state_dict(), artifacts_dir / "m_unlearn_final.pt")
    with (artifacts_dir / "metrics.json").open("w") as f:
        json.dump({"before": before, "after": after, "training_history": result.metrics["history"]}, f, indent=2)

    print(f"Done. M_unlearn artifacts written to {artifacts_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/cifar100_fl_colab.yaml")
    parser.add_argument("--unlearning-config", default="configs/unlearning.yaml")
    args = parser.parse_args()
    main(args.base_config, args.unlearning_config)
