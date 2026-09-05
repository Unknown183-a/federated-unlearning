# Phase 05 — Gradient Ascent

**Status:** Code complete, unit-tested; not yet run against real CIFAR-100
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
evaluates test/forget-client accuracy before and after. Smoke-tested
end-to-end on synthetic data (checkpoint load → unlearn → evaluate all
run cleanly); not yet run against real CIFAR-100 — pending a GPU run,
same as Phase 04.

Next: run `experiments/run_gradient_ascent.py`, then compare
`M_unlearn`'s test/forget-client accuracy against `M_old` (60.02%/n/a)
and `M_retrain` (70.89%/65.30%) to judge whether gradient ascent
approximates full retraining's forgetting effect without its cost.
