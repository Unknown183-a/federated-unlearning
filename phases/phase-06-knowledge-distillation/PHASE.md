# Phase 06 — Knowledge Distillation

**Status:** Done — real result obtained (reveals why Phase 07 must interleave, not sequence, GA+KD)
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
engine.

**Result (GPU, Colab, real CIFAR-100):**

| Stage | Test acc. | Forget-client acc. |
|---|---:|---:|
| M_old | 60.02% | 67.10% |
| M_unlearn (Phase 05, GA only) | 55.41% | 52.31%* |
| + sequential KD (this phase) | 59.99% | 66.95% |

*(52.31% here vs. 52.07% reported in Phase 05 — trivial run-to-run
variance, same setup.)*

**Important finding:** sequential KD repaired overall test accuracy
almost perfectly (55.41% → 59.99%, ~M_old level) — but it also undid
nearly all of Gradient Ascent's forgetting (52.31% → 66.95%, back near
M_old's 67.10%). This is not a bug; it's the expected consequence of
running the two steps *sequentially*. Pure KD's only training signal
is "match the teacher on remaining-client data," with no explicit
instruction to avoid restoring forget-client knowledge — and since
CIFAR-100's non-IID Dirichlet split doesn't cleanly separate classes
between clients, matching M_old's behavior on remaining data appears
to indirectly restore behavior on overlapping classes the forget
client also had. **This is precisely why Phase 07's `UnlearningEngine`
must interleave GA and KD within the same training loop rather than
running them as two separate stages** — simultaneous optimization
should let the two objectives balance against each other, rather than
one (KD) completely overwriting the other's (GA's) effect. This result
is good evidence for that design choice, not a failure of Phase 06's
implementation — `KnowledgeDistiller.distill()` did exactly its job.
