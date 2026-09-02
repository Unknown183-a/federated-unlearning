# Phase 04 — Full Retraining Baseline

**Status:** Code complete; not yet run for a real result
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
`run_full_retraining` reuses `FederatedServer` exactly as specced, is
smoke-tested end-to-end on synthetic data, and writes
`m_retrain_final.pt` + `metrics.csv` + `forget_client_metrics.json` to
`artifacts/experiments/full_retraining/`. Not yet run against the real
CIFAR-100 data — run `experiments/run_cifar100_fl.py` first to produce
a real `M_old`, then `experiments/run_full_retraining.py` for `M_retrain`.
