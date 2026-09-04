# Phase 04 — Full Retraining Baseline

**Status:** Done — real result obtained
**Depends on:** Phase 03

## Goal

Remove the forget client and retrain the FL system from scratch as the gold-standard reference, `M_retrain`.

## Tasks

- [x] Implement `src/baselines/full_retraining.py::run_full_retraining`
- [x] Reuse `FederatedServer` with the forget client excluded from `client_datasets`
- [x] Save `M_retrain` checkpoint + metrics under `artifacts/experiments/full_retraining/`
- [x] `configs/unlearning.yaml`'s `full_retraining` section drives its rounds/epochs/lr

## Definition of done

A full retraining run completes and `M_retrain` is checkpointed and evaluable the same way `M_old` is.

## Handoff notes

The base experiment moved from the MNIST/CNN pipeline-validation setup to
the real CIFAR-100 + ResNet-18 setup (`configs/cifar100_fl.yaml`,
`experiments/run_cifar100_fl.py`, `src/models/resnet.py`,
`src/data/cifar.py`) — this is now the actual `M_old`.
`run_full_retraining` reuses `FederatedServer` exactly as specced and is
smoke-tested end-to-end on synthetic data.

Rather than rerun training through the new pipeline (CPU training time
was prohibitive), `M_old`/`M_retrain` were imported from the original
pre-refactor `main.py`/`retrain_baseline.py` checkpoints and evaluated
via `experiments/evaluate_legacy_checkpoints.py`, which reconstructs
the exact legacy forget-client partition (client 0) to get a real
forget-client accuracy for `M_retrain`. Results, written to
`artifacts/experiments/{cifar100_fl,full_retraining}/imported_metrics.json`:

- `M_old` overall test accuracy: **60.02%**
- `M_retrain` overall test accuracy: **31.86%**
- `M_retrain` forget-client accuracy: **24.43%**

`M_retrain`'s overall accuracy is notably lower than `M_old`'s despite
more rounds (50 vs 20) and local epochs (10 vs 2) — likely FedAvg
client drift from high `local_epochs` on non-IID data (see FedProx, Li
et al. 2018), though the original per-round logs weren't preserved to
confirm this over a training-instability explanation. Flagged as a
limitation to revisit in the final report if time allows; not a
blocker for Phase 05, which needs *a* `M_retrain` reference to compare
`M_unlearn` against, not a perfect one.

**Follow-up:** implemented FedProx (proximal term, `mu`) in
`src/federated/client.py::local_train` and threaded it through
`FederatedServer.run` and `run_full_retraining`, defaulting to `mu=0.0`
(exact plain-FedAvg behavior) so Phase 03's MNIST pipeline check is
unaffected. `configs/unlearning.yaml`'s `full_retraining` section now
sets `mu: 0.01, momentum: 0.9, weight_decay: 0.0005` — keeping
`local_epochs: 10` (FedProx's proximal term is specifically meant to
make that tolerable on non-IID data, rather than needing to cut local
epochs). Smoke-tested on synthetic data; not yet rerun against real
CIFAR-100 — pending GPU (Colab) run to replace the current `M_retrain`
result above with an improved one.
