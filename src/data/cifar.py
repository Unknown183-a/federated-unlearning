"""CIFAR-100 loading and normalization for the real (non-pipeline-validation) FL experiment."""
from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset
from torchvision import datasets, transforms

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_cifar100_datasets(data_dir: str | Path) -> tuple[Dataset, Dataset]:
    """Download (if needed) and return the CIFAR-100 train/test datasets.

    The train set gets light augmentation (random crop + horizontal flip);
    the test set does not, matching standard CIFAR practice.
    """
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)]
    )

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    train_set = datasets.CIFAR100(root=str(data_dir), train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR100(root=str(data_dir), train=False, download=True, transform=test_transform)
    return train_set, test_set
