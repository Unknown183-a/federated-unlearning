"""Run Phase 06: Knowledge Distillation.

Standalone KD doesn't produce a new "M_x" worth comparing on its own —
its real role is repairing the collateral damage Gradient Ascent causes
(Phase 05: M_unlearn lost 4.6 points of overall test accuracy). This
runner demonstrates that repair directly: student = M_unlearn (the
GA-damaged model), teacher = M_old, distilled on the remaining
(non-forget) clients' data. Phase 07's UnlearningEngine will combine
both steps into a single run.

Usage:
    python experiments/run_knowledge_distillation.py \
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
from torch.utils.data import DataLoader, ConcatDataset

from src.data.cifar import get_cifar100_datasets
from src.data.partition import partition_dataset
from src.federated.client import FederatedClient
from src.models.resnet import build_resnet18
from src.unlearning.forget_client import remaining_clients
from src.unlearning.knowledge_distillation import KnowledgeDistiller
from src.utils.config import load_config, save_config


def main(base_config_path: str, unlearning_config_path: str) -> None:
    base_cfg = load_config(base_config_path)
    unlearn_cfg = load_config(unlearning_config_path)
    torch.manual_seed(unlearn_cfg.seed)

    artifacts_dir = Path(unlearn_cfg.output.artifacts_dir) / "knowledge_distillation"
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
    remaining = remaining_clients(client_datasets, forget_client_id)
    remaining_dataset = ConcatDataset(list(remaining.values()))

    kd_cfg = unlearn_cfg.knowledge_distillation
    remaining_loader = DataLoader(remaining_dataset, batch_size=base_cfg.data.batch_size, shuffle=True)

    teacher = build_resnet18(num_classes=base_cfg.model.num_classes).to(device)
    teacher.load_state_dict(torch.load(unlearn_cfg.base_checkpoint, map_location=device))

    ga_checkpoint = Path(unlearn_cfg.output.artifacts_dir) / "gradient_ascent" / "m_unlearn_final.pt"
    student_init = build_resnet18(num_classes=base_cfg.model.num_classes).to(device)
    student_init.load_state_dict(torch.load(ga_checkpoint, map_location=device))

    test_client = FederatedClient(-1, test_set, batch_size=base_cfg.data.batch_size, device=device)
    forget_client = FederatedClient(
        forget_client_id, client_datasets[forget_client_id], batch_size=base_cfg.data.batch_size, device=device
    )

    before = {"test": test_client.evaluate(student_init), "forget": forget_client.evaluate(student_init)}
    print("Before distillation (= M_unlearn from Phase 05):", before)

    distiller = KnowledgeDistiller(teacher=teacher, student=student_init, config={**kd_cfg, "device": device})
    m_distilled = distiller.distill(remaining_loader)

    after = {"test": test_client.evaluate(m_distilled), "forget": forget_client.evaluate(m_distilled)}
    print("After distillation:", after)

    torch.save(m_distilled.state_dict(), artifacts_dir / "m_distilled_final.pt")
    with (artifacts_dir / "metrics.json").open("w") as f:
        json.dump({"before": before, "after": after}, f, indent=2)

    print(f"Done. Artifacts written to {artifacts_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/cifar100_fl.yaml")
    parser.add_argument("--unlearning-config", default="configs/unlearning.yaml")
    args = parser.parse_args()
    main(args.base_config, args.unlearning_config)
