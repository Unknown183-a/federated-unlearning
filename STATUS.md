# Status

Source of truth for where the project actually stands. Kept in sync
with `README.md`'s Current Phase checklist above the fold — update
both when a phase finishes.

## Current phase

**Phase 05 — Gradient Ascent** (not started)

Phases 00-04 are complete with real results. The base experiment moved
from the MNIST/CNN pipeline-validation setup to the real CIFAR-100 +
ResNet-18 setup. M_old was imported from a pre-refactor `main.py`
checkpoint and evaluated through the new pipeline
(`experiments/evaluate_legacy_checkpoints.py`). M_retrain was
retrained from scratch through the new pipeline
(`experiments/run_full_retraining.py`) on GPU (Colab), using FedProx
(proximal term) after an initial plain-FedAvg attempt showed severe
client drift — see history below. Phase 08 (Evaluation Framework) is
partially done ahead of schedule — accuracy/cost/comparison utilities
exist since they only depend on Phase 03, not on the unlearning engine.

**Results** (CIFAR-100, 5 clients, non-IID Dirichlet α=0.5, forget client = 0):

| Model                        | Overall test acc. | Forget-client acc. | Rounds × local epochs |
|-------------------------------|-------------------:|--------------------:|------------------------|
| M_old                         | 60.02%             | n/a (unseeded partition, not recoverable) | 20 × 2 |
| **M_retrain (FedProx, final)**| **70.89%**          | **65.30%**           | 50 × 10, μ=0.01 |
| ~~M_retrain (plain FedAvg)~~  | ~~31.86%~~ (superseded) | ~~24.43%~~ (superseded) | 50 × 10, μ=0 |

**History:** the first M_retrain attempt (plain FedAvg, imported from
a pre-refactor `retrain_baseline.py` checkpoint) scored notably lower
than M_old despite more rounds/local epochs — diagnosed as FedAvg
client drift from `local_epochs=10` on non-IID data (see FedProx, Li
et al. 2018). Fixed by implementing FedProx's proximal term
(`mu=0.01`) in `src/federated/client.py::local_train`, plus adding
momentum/weight_decay, and rerunning `run_full_retraining.py` on GPU.
The new M_retrain (70.89%) now exceeds M_old (60.02%), and the
forget-client accuracy (65.30%) is sensibly close to but below the
overall accuracy, consistent with never having trained on that
client's data. This is now the reference M_retrain for Phase 05-07.

```
Data pipeline      ██████████ 100%
FL baseline        ██████████ 100%
Evaluation utils   ████░░░░░░  40%   (accuracy/cost/comparison done; forgetting/MIA blocked on unlearning)
Full retraining    ██████████ 100%
Gradient ascent    ░░░░░░░░░░   0%
Knowledge distill. ░░░░░░░░░░   0%
Unlearning engine  ░░░░░░░░░░   0%
MIA                ░░░░░░░░░░   0%
Experiments/report ░░░░░░░░░░   0%
```

## Phase checklist

- [x] Phase 00 — Repo & Skeleton
- [x] Phase 01 — Data Pipeline
- [x] Phase 02 — Model + FedAvg
- [x] Phase 03 — Initial FL Experiment (real result: CIFAR-100+ResNet-18, M_old = 60.02% test acc.)
- [x] Phase 04 — Full Retraining Baseline (real result: M_retrain = 70.89% test acc., 65.30% forget-client acc., via FedProx — see history above)
- [ ] Phase 05 — Gradient Ascent
- [ ] Phase 06 — Knowledge Distillation
- [ ] Phase 07 — Unlearning Engine
- [ ] Phase 08 — Evaluation Framework (partial — see above)
- [ ] Phase 09 — MIA
- [ ] Phase 10 — Controlled Experiments
- [ ] Phase 11 — Plots + Reports
- [ ] Phase 12 — Final Report / Viva

## Next action

Start Phase 05 (Gradient Ascent) in
`phases/phase-05-gradient-ascent/PHASE.md`, using
`artifacts/experiments/cifar100_fl/imported_original_model.pt` as M_old
and `artifacts/experiments/full_retraining/m_retrain_final.pt` as the
M_retrain reference to compare M_unlearn against.
