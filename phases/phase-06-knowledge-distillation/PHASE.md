# Phase 06 — Knowledge Distillation

**Status:** Code complete, unit-tested; not yet run against real CIFAR-100
**Depends on:** Phase 03

## Goal

Implement Knowledge Distillation using `M_old` as teacher, to preserve knowledge from remaining clients while the model is being unlearned.

## Tasks

- [x] Implement `src/unlearning/knowledge_distillation.py::KnowledgeDistiller.distill`
- [x] Use `kd_loss` from `src/unlearning/losses.py` (temperature configurable)
- [x] `configs/unlearning.yaml` → `knowledge_distillation` drives temperature/epochs/lr/lambda
- [x] Unit test: student accuracy on remaining clients doesn't collapse after distillation

## Definition of done

`KnowledgeDistiller.distill()` runs without error and remaining-client accuracy stays close to `M_old`'s.

## Handoff notes

Two unit tests in `tests/test_knowledge_distillation.py`: one confirms
a student recovers toward the teacher's accuracy on remaining-client
data (tuned to `lr=0.1, epochs=30, temperature=2.0` for the small
synthetic setup — worth noting a *higher* lr (0.3) actually converged
worse, likely from unstable steps against the soft KL-divergence
target), the other confirms neither teacher nor the original student
object gets mutated (matches the existing scaffold's `copy.deepcopy`
pattern). All 10 repo tests pass.

Added `experiments/run_knowledge_distillation.py` (not explicitly
required by the task list, but needed to show something concrete):
takes Phase 05's `M_unlearn` (GA-damaged) as the student and `M_old` as
the teacher, distills on the remaining (non-forget) clients' real
data, and evaluates test/forget-client accuracy before/after. This
directly demonstrates KD's intended role — repairing the collateral
damage Gradient Ascent caused to overall accuracy (Phase 05: M_unlearn
lost 4.6 points vs. M_old) — ahead of Phase 07 combining both into one
engine. Smoke-tested end-to-end on synthetic data; not yet run against
real CIFAR-100 — pending a GPU run, same pattern as Phases 04-05.
