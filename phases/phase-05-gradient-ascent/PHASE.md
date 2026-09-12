# Phase 05 — Gradient Ascent

**Status:** Done — real result obtained
**Depends on:** Phase 03

## Goal

Implement the Gradient Ascent unlearner: push the model away from fitting the forget client's data.

## Tasks

- [x] Implement `src/unlearning/gradient_ascent.py::GradientAscentUnlearner.unlearn`
- [x] Use `gradient_ascent_loss` from `src/unlearning/losses.py`
- [x] lr, epochs, batch size all config-driven (`configs/unlearning.yaml` → `gradient_ascent`)
- [x] Unit test: forget-client loss increases after `unlearn()`

## Definition of done

`GradientAscentUnlearner.unlearn()` runs without error and measurably increases loss on the forget client's data.

## Handoff notes

Implemented with gradient clipping (`max_grad_norm`, default 1.0,
added to `configs/unlearning.yaml`) — without it, the unbounded ascent
objective (minimizing negative cross-entropy has no lower bound)
diverged to NaN within a handful of steps during testing. Two unit
tests added in `tests/test_gradient_ascent.py`: one confirms loss
measurably increases (per Definition of Done), the other confirms
`unlearn()` doesn't mutate the original model in place (it operates on
an internal `copy.deepcopy`, per the existing scaffold). All 8 repo
tests pass.

Added `experiments/run_gradient_ascent.py` (not explicitly required by
the task list above, but needed to actually produce a comparable
`M_unlearn`): loads `M_old` from `configs/unlearning.yaml`'s
`base_checkpoint`, runs ascent on the forget client's real data, and
evaluates test/forget-client accuracy before and after.

**Result (GPU, Colab, real CIFAR-100):**

- `M_old` (before unlearning): 60.02% test acc., 67.10% forget-client acc.
- `M_unlearn` (after 5 epochs of ascent): 55.41% test acc., 52.07% forget-client acc.

Forget-client accuracy dropped 15.0 points vs. only 4.6 points on
overall test accuracy — Gradient Ascent forgets client 0 faster and
far more cheaply than full retraining (5 epochs on one client vs. 50
rounds × 10 local epochs across four), but with real collateral
damage: it also degrades unrelated performance, and its forgetting is
less thorough than `M_retrain`'s (52.07% vs. 65.30% forget-client
acc. — counterintuitively *lower* than M_retrain's, since M_retrain
never learned that data at all under a from-scratch fit that also
improved overall, whereas GA is actively editing a model that already
knew it). This cost/precision tradeoff is the expected weakness of
plain gradient ascent from the literature, and motivates Phase 06's
Knowledge Distillation approach as a potential improvement.

Artifacts: `artifacts/experiments/unlearning_ga_kd/gradient_ascent/`
(`m_unlearn_final.pt`, `metrics.json`).
