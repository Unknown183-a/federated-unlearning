# Phase 03 — Initial FL Experiment

**Status:** Done — real result obtained (scope changed from original plan, see Handoff notes)
**Depends on:** Phase 02

## Goal

Wire client + FedAvg into a full server loop and produce the initial pipeline-validation experiment: 5 clients, 50 rounds, 10 local epochs/round.

## Tasks

- [x] `src/federated/server.py` — round loop, checkpointing, per-round metrics
- [x] `src/federated/round_manager.py` — metrics history + CSV export
- [x] `src/utils/checkpoint.py`, `src/utils/logging.py`, `src/utils/config.py`, `src/utils/seed.py`
- [x] `experiments/run_initial_fl.py` — runnable end-to-end script (MNIST/CNN pipeline check)
- [x] `configs/initial_fl.yaml`
- [x] Produce a real `M_old` and commit its artifacts

## Definition of done

`python experiments/run_initial_fl.py` completes 50 rounds and writes config.yaml, metrics.csv, training.log, partition_metadata.json, and checkpoints/round_*.pt.

## Handoff notes

**Scope changed from the original plan, on purpose, before Phase 04.**
`run_initial_fl.py`/`configs/initial_fl.yaml` above were the original
MNIST + simple-CNN pipeline-validation experiment — and were genuinely
never run; that specific deliverable was superseded, not completed as
originally scoped.

Instead, CIFAR-100 + ResNet-18 became the real experiment going
forward (decided at the start of Phase 04, once it became clear that
was the intended dataset for the actual thesis). This produces the
real `M_old` via `configs/cifar100_fl.yaml` +
`experiments/run_cifar100_fl.py` (both added during Phase 04, not part
of this phase's original scaffold).

The actual `M_old` checkpoint used throughout Phases 04-07
(`artifacts/experiments/cifar100_fl/imported_original_model.pt`,
committed to the repo directly) was imported from a pre-refactor local
run rather than produced by `run_cifar100_fl.py` itself, since it
already existed and its original training used an unseeded partition
that can't be exactly reproduced — see that file's directory and
`docs/results.md` footnote 1 for the caveat. Real result: **60.02%
test accuracy** (`artifacts/experiments/cifar100_fl/imported_metrics.json`).

`run_initial_fl.py`/`configs/initial_fl.yaml` remain in the repo as a
working, tested MNIST pipeline sanity-check (useful for quickly
verifying the FL loop still works without a full CIFAR-100 run), but
they are not part of the thesis's real result chain.
