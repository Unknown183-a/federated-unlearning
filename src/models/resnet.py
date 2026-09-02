"""ResNet-18 adapted for 32x32 CIFAR images — the real (non-pipeline-validation) model.

Swaps the initial 7x7/stride-2 conv + maxpool (designed for 224x224
ImageNet inputs) for a 3x3/stride-1 conv and no maxpool, the standard
adaptation for 32x32 CIFAR inputs.
"""
from __future__ import annotations

import torch.nn as nn
import torchvision


def build_resnet18(num_classes: int = 100) -> nn.Module:
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
