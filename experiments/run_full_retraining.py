"""Run the Phase 04 full-retraining baseline: M_retrain.

M_retrain is trained on the same CIFAR-100 client partition as M_old
(configs/cifar100_fl.yaml), with the forget client excluded, using the
schedule from configs/unlearning.yaml's `full_retraining` section.

Usage:
    python experiments/run_full_retraining.py \
        --base-config configs/cifar100_fl.yaml \
        --unlearning-config configs/unlearning.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines.full_retraining import run_full_retraining
from src.data.cifar import get_cifar100_datasets
from src.data.partition import partition_dataset
from src.data.utils import save_partition_metadata
from src.models.resnet import build_resnet18
from src.utils.config import load_config, save_config
from src.utils.seed import set_seed


def main(base_config_path: str, unlearning_config_path: str) -> None:
    base_cfg = load_config(base_config_path)
    unlearn_cfg = load_config(unlearning_config_path)
    set_seed(unlearn_cfg.seed)

    # PHASE.md: "Save M_retrain checkpoint + metrics under
    # artifacts/experiments/full_retraining/" — a sibling of the GA/KD
    # unlearning_ga_kd artifacts dir, not nested inside it.
    artifacts_dir = Path(unlearn_cfg.output.artifacts_dir).parent / "full_retraining"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    save_config(
        {"base_config": base_config_path, "unlearning_config": unlearning_config_path},
        artifacts_dir / "config.yaml",
    )

    train_set, test_set = get_cifar100_datasets(base_cfg.data.data_dir)
    client_datasets = partition_dataset(
        train_set,
        num_clients=base_cfg.data.num_clients,
        strategy=base_cfg.data.partition,
        alpha=base_cfg.data.get("non_iid_alpha", 0.5),
        seed=base_cfg.seed,
    )
    save_partition_metadata(client_datasets, artifacts_dir / "partition_metadata.json")

    def model_factory():
        return build_resnet18(num_classes=base_cfg.model.num_classes)

    run_full_retraining(
        client_datasets=client_datasets,
        forget_client_id=unlearn_cfg.forget_client_id,
        model_factory=model_factory,
        config={
            **unlearn_cfg.full_retraining,
            "batch_size": base_cfg.data.batch_size,
            "device": base_cfg.device,
            "artifacts_dir": artifacts_dir,
            "save_every_n_rounds": base_cfg.output.save_every_n_rounds,
            "test_dataset": test_set,
        },
    )
    print(f"Done. M_retrain artifacts written to {artifacts_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/cifar100_fl.yaml")
    parser.add_argument("--unlearning-config", default="configs/unlearning.yaml")
    args = parser.parse_args()
    main(args.base_config, args.unlearning_config)
