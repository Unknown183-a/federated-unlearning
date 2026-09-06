"""Unlearning engine: jointly optimizes Gradient Ascent + Knowledge Distillation.

Per docs/methodology.md's combined objective:

    L_total = lambda_forget * L_forget + lambda_kd * L_KD

computed jointly, in a single training loop over paired batches from
the forget client and remaining clients — NOT by running
GradientAscentUnlearner then KnowledgeDistiller sequentially. Phase 06
showed empirically that sequential application undoes almost all of
GA's forgetting once KD runs afterward (KD's only signal is "match the
teacher," with nothing telling it to preserve GA's forgetting). Joint
optimization lets the two objectives balance against each other at
every step instead.
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn

from .losses import gradient_ascent_loss, kd_loss


@dataclass
class UnlearningResult:
    model: nn.Module
    metrics: Dict[str, Any]


class UnlearningEngine:
    def __init__(self, m_old: nn.Module, config: Dict[str, Any]):
        self.m_old = m_old
        self.config = config

    def run(self, forget_client_id: int, forget_loader, remaining_loader) -> UnlearningResult:
        """Jointly forget `forget_loader` and preserve `remaining_loader` in one loop.

        config keys (from configs/unlearning.yaml's `knowledge_distillation`
        section, which already carries lambda_forget/lambda_kd/temperature
        anticipating this joint use): lr, epochs, temperature, lambda_forget,
        lambda_kd. Optional: device (default "cpu"), max_grad_norm (default
        1.0 — the forgetting term is an unbounded objective, same issue as
        in GradientAscentUnlearner).

        The teacher (`self.m_old`) is used read-only and never updated; the
        student starts as a fresh copy of it. Each step pairs one batch from
        `forget_loader` with one from `remaining_loader`, cycling whichever
        is shorter (forget-client data is typically much smaller) so every
        batch of the longer loader gets a paired batch each epoch.
        """
        device = self.config.get("device", "cpu")
        teacher = self.m_old
        teacher.to(device)
        teacher.eval()

        student = copy.deepcopy(self.m_old).to(device)
        student.train()

        optimizer = torch.optim.SGD(student.parameters(), lr=self.config["lr"])
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        lambda_forget = self.config.get("lambda_forget", 1.0)
        lambda_kd = self.config.get("lambda_kd", 1.0)
        temperature = self.config.get("temperature", 2.0)
        epochs = self.config["epochs"]

        n_forget, n_remaining = len(forget_loader), len(remaining_loader)
        history = []

        for epoch in range(epochs):
            forget_iter = itertools.cycle(forget_loader) if n_forget < n_remaining else iter(forget_loader)
            remaining_iter = itertools.cycle(remaining_loader) if n_remaining < n_forget else iter(remaining_loader)

            epoch_forget_loss, epoch_kd_loss, steps = 0.0, 0.0, 0
            for (xf, yf), (xr, _) in zip(forget_iter, remaining_iter):
                xf, yf, xr = xf.to(device), yf.to(device), xr.to(device)

                with torch.no_grad():
                    teacher_logits = teacher(xr)

                forget_l = gradient_ascent_loss(student(xf), yf)
                kd_l = kd_loss(student(xr), teacher_logits, temperature=temperature)
                loss = lambda_forget * forget_l + lambda_kd * kd_l

                optimizer.zero_grad()
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
                optimizer.step()

                epoch_forget_loss += forget_l.item()
                epoch_kd_loss += kd_l.item()
                steps += 1

            history.append(
                {
                    "epoch": epoch + 1,
                    "forget_loss": epoch_forget_loss / max(steps, 1),
                    "kd_loss": epoch_kd_loss / max(steps, 1),
                }
            )

        metrics = {"forget_client_id": forget_client_id, "history": history}
        return UnlearningResult(model=student, metrics=metrics)
