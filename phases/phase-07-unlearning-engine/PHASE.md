# Phase 07 — Unlearning Engine

**Status:** Code complete, unit-tested; not yet run against real CIFAR-100
**Depends on:** Phase 05, Phase 06

## Goal

Combine Gradient Ascent + Knowledge Distillation into the end-to-end unlearning pipeline that produces `M_unlearn`.

## Tasks

- [x] Implement `src/unlearning/engine.py::UnlearningEngine.run`
- [x] ~~Calls `GradientAscentUnlearner` then `KnowledgeDistiller` per blueprint Section 9~~ — see Handoff notes: implemented as a **joint** loss instead, per `docs/methodology.md`
- [x] Save `M_unlearn` checkpoint + metrics under `artifacts/experiments/unlearning_ga_kd/`
- [x] Use `src/unlearning/forget_client.py` for client selection (already implemented)

## Definition of done

`UnlearningEngine.run()` returns an `UnlearningResult` with a checkpointed `M_unlearn` and recorded metrics.

## Handoff notes

**Deviated from this file's original task wording ("calls GA then KD")
on purpose, with evidence.** Phase 06 empirically showed that running
GA then KD *sequentially* undoes almost all of GA's forgetting: KD's
only signal is "match the teacher on remaining-client data," and under
non-IID class overlap between clients, that ends up restoring
forget-client behavior too (52.31% → 66.95% forget-client accuracy —
back near M_old's 67.10%). No "blueprint" document with numbered
sections exists anywhere in this repo's git history (checked `git log
--all`) — but `docs/methodology.md` does specify the actual intended
objective explicitly:

    L_total = lambda_forget * L_forget + lambda_kd * L_KD

a single **combined** loss, not two sequential stages. Implemented
`UnlearningEngine.run()` accordingly: one training loop, each step
pairs a batch from the forget client with a batch from the remaining
clients, computes both losses against the same student, combines them
with `lambda_forget`/`lambda_kd`, and does one backward pass. The
student starts as a copy of `M_old` (teacher); `M_old` itself is used
read-only and never mutated. Gradient clipping (`max_grad_norm`,
reused from the `gradient_ascent` config section) prevents the
unbounded forgetting term from diverging, same issue as Phase 05.

Three unit tests in `tests/test_unlearning_engine.py`. The important
one, `test_joint_engine_forgets_without_collapsing_remaining_accuracy`,
specifically checks the property Phase 06 showed sequential
application fails at: forget-loss increases AND remaining-client
accuracy doesn't collapse, in the *same* run. All 13 repo tests pass.

Added `experiments/run_unlearning_engine.py`: loads `M_old`, builds
forget/remaining loaders from the real partition, runs the joint
engine, evaluates test/forget-client accuracy before/after. Reuses
`knowledge_distillation`'s config section for `lr`/`epochs`/
`temperature`/`lambda_forget`/`lambda_kd` (it already carried
`lambda_forget` unused in Phase 06, clearly anticipating this engine)
and `gradient_ascent`'s `batch_size`/`max_grad_norm` for the forget
side. Smoke-tested end-to-end on synthetic data; not yet run against
real CIFAR-100 — pending a GPU run, same pattern as Phases 04-06.
