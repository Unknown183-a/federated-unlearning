"""Knowledge Distillation module (teacher = M_old, student = model being unlearned).

Trains the student to match the teacher's soft output distribution on
the *remaining* (non-forget) clients' data — the opposite objective
from Gradient Ascent's forgetting loss. Standalone here, this measures
whether pure distillation preserves remaining-client knowledge; Phase
07's UnlearningEngine combines this with GradientAscentUnlearner so a
single run both forgets and preserves.
"""
from __future__ import annotations

import copy
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .losses import kd_loss


class KnowledgeDistiller:
    def __init__(self, teacher: nn.Module, student: nn.Module, config: Dict[str, Any]):
        self.teacher = teacher
        self.student = copy.deepcopy(student)
        self.config = config

    def distill(self, remaining_loader: DataLoader) -> nn.Module:
        """Train the student toward the teacher's soft outputs on `remaining_loader`.

        config keys: lr, epochs, temperature (required, from
        configs/unlearning.yaml's `knowledge_distillation` section);
        lambda_kd (optional, defaults to 1.0, scales the KD loss —
        Phase 07's engine combines this with a forgetting loss term
        using its own lambda_forget, so this stays a clean multiplier
        rather than baking in a fixed combination here); device
        (optional, defaults to "cpu").

        Only inputs from `remaining_loader` are used — labels are
        ignored, since distillation targets are the teacher's own
        outputs, not the ground-truth labels.
        """
        device = self.config.get("device", "cpu")
        self.teacher.to(device)
        self.student.to(device)
        self.teacher.eval()
        self.student.train()

        optimizer = torch.optim.SGD(self.student.parameters(), lr=self.config["lr"])
        temperature = self.config.get("temperature", 2.0)
        lambda_kd = self.config.get("lambda_kd", 1.0)

        for _ in range(self.config["epochs"]):
            for x, _ in remaining_loader:
                x = x.to(device)
                with torch.no_grad():
                    teacher_logits = self.teacher(x)
                student_logits = self.student(x)

                loss = lambda_kd * kd_loss(student_logits, teacher_logits, temperature=temperature)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self.student
