# Status

Source of truth for where the project actually stands. Kept in sync
with `README.md`'s Current Phase checklist above the fold — update
both when a phase finishes.

## Current phase

**Phase 05 — Gradient Ascent** (not started)

Phases 00-04 are complete with real results. The base experiment moved
from the MNIST/CNN pipeline-validation setup to the real CIFAR-100 +
ResNet-18 setup. M_old and M_retrain checkpoints were produced by the
original (pre-refactor) `main.py`/`retrain_baseline.py` scripts and
then imported + evaluated through the new pipeline
(`experiments/evaluate_legacy_checkpoints.py`) rather than rerun, since
CPU training time was prohibitive; see caveat below. Phase 08
(Evaluation Framework) is partially done ahead of schedule —
accuracy/cost/comparison utilities exist since they only depend on
Phase 03, not on the unlearning engine.

**Results so far** (CIFAR-100, 5 clients, non-IID Dirichlet α=0.5, forget client = 0):

| Model      | Overall test acc. | Forget-client acc. | Rounds × local epochs |
|------------|-------------------:|--------------------:|------------------------|
| M_old      | 60.02%             | n/a (unseeded partition, not recoverable) | 20 × 2 |
| M_retrain  | 31.86%             | 24.43%              | 50 × 10 |

**Caveat:** M_retrain's overall accuracy is notably *lower* than M_old's
despite more rounds/local epochs. Likely cause: `local_epochs=10` on
non-IID (α=0.5) data is a known FedAvg failure mode — clients drift far
toward their own local optima before each aggregation, and simple
averaging can't reconcile that (see FedProx, Li et al. 2018). Original
per-round training logs from the pre-refactor run weren't preserved, so
this can't be distinguished from a training-instability issue without
rerunning — flagged here as a limitation to revisit in the final report
if time allows, not silently smoothed over.

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
- [x] Phase 04 — Full Retraining Baseline (real result: M_retrain = 31.86% test acc., 24.43% forget-client acc. — see caveat above)
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
and `artifacts/experiments/full_retraining/imported_retrained_model.pt`
as the M_retrain reference to compare M_unlearn against.
