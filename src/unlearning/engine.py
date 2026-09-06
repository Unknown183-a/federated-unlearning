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
from typing import Any, Dict, List

import torch
import torch.nn as nn

from .losses import gradient_ascent_loss, kd_loss


def _clip_grads_by_norm(grads: List[torch.Tensor], max_norm: float) -> List[torch.Tensor]:
    """Scale a list of gradient tensors so their combined L2 norm <= max_norm.

    Unlike torch.nn.utils.clip_grad_norm_, this operates on plain gradient
    tensors (not parameters with a populated .grad), so it can be applied
    to the forgetting and KD gradients independently before they're
    combined into a single update.
    """
    total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        grads = [g * scale for g in grads]
    return grads


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
        params = list(student.parameters())

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

                # Backprop each objective separately and clip each
                # gradient independently BEFORE combining. Clipping the
                # combined sum instead would let the unbounded ascent
                # term's raw gradient magnitude dominate the update
                # direction regardless of lambda weighting, since
                # clipping only rescales magnitude, not per-term balance.
                forget_l = gradient_ascent_loss(student(xf), yf)
                forget_grads = torch.autograd.grad(forget_l, params, retain_graph=True, allow_unused=True)
                forget_grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(forget_grads, params)]

                kd_l = kd_loss(student(xr), teacher_logits, temperature=temperature)
                kd_grads = torch.autograd.grad(kd_l, params, allow_unused=True)
                kd_grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(kd_grads, params)]

                if max_grad_norm is not None:
                    forget_grads = _clip_grads_by_norm(forget_grads, max_grad_norm)
                    kd_grads = _clip_grads_by_norm(kd_grads, max_grad_norm)

                optimizer.zero_grad()
                for p, fg, kg in zip(params, forget_grads, kd_grads):
                    p.grad = lambda_forget * fg + lambda_kd * kg
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
