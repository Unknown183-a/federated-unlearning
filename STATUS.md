# Status

Source of truth for where the project actually stands. Kept in sync
with `README.md`'s Current Phase checklist above the fold — update
both when a phase finishes.

## Current phase

**Phase 08 — Evaluation Framework** (partial — see below)

Phases 00-07 are complete with real results, all against the real
CIFAR-100 + ResNet-18 setup. M_old was imported from a pre-refactor
checkpoint; M_retrain was retrained via FedProx on GPU (Colab);
Gradient Ascent, sequential KD, and the final joint GA+KD engine were
all run for real on GPU. accuracy/cost/comparison utilities in Phase 08
already exist and are in active use (`build_comparison_table` generates
`artifacts/experiments/comparison_table.csv`); forgetting/MIA metrics
are still blocked on nothing now that the unlearning engine exists —
MIA itself is Phase 09's job.

**Results** (CIFAR-100, 5 clients, non-IID Dirichlet α=0.5, forget client = 0):

| Model                                  | Overall test acc. | Forget-client acc. | Cost |
|------------------------------------------|-------------------:|---------------------:|------|
| M_old                                     | 60.02%             | 67.10%                | 20 rounds × 2 local epochs |
| M_retrain (FedProx, gold standard)        | 70.89%             | 65.30%                | 50 rounds × 10 local epochs, μ=0.01 |
| M_unlearn (Gradient Ascent only)          | 55.41%             | 52.31%                | 5 epochs on forget client only |
| M_unlearn + sequential KD                 | 59.99%             | 66.95%                | + 5 epochs KD on remaining clients |
| **M_unlearn (joint GA+KD engine, final)** | **57.78%**          | **57.36%**             | 5 epochs joint, λ_forget=1.0, λ_kd=1.0 |

**Interpretation:** Gradient Ascent forgot client 0's data faster and
far cheaper than full retraining, but less precisely: forget-client
accuracy dropped 15.0 points (67.10% → 52.31%) while overall test
accuracy also dropped 4.6 points (60.02% → 55.41%) — collateral damage
to unrelated knowledge, a known weakness of plain gradient ascent. By
contrast, M_retrain improved on *both* axes relative to M_old, since
it's a from-scratch fit rather than a targeted edit.

**Phase 06 finding:** applying KD *sequentially* after GA repaired
overall accuracy almost perfectly (55.41% → 59.99%, ~M_old level) — but
also undid nearly all of the forgetting (52.31% → 66.95%, back near
M_old's 67.10%). This showed the two objectives can't be run as
separate stages — KD's only signal is "match the teacher," with nothing
telling it to preserve GA's forgetting.

**Phase 07 result:** implemented `UnlearningEngine` per
`docs/methodology.md`'s combined loss (`L_total = λ_forget·L_forget +
λ_kd·L_KD`), computed jointly each step rather than sequentially. First
attempt had a real bug — clipping the *combined* gradient let the
unbounded ascent term dominate regardless of λ weighting, producing a
result worse than plain GA on both axes (50.00%/38.13%). Fixed by
clipping each loss's gradient independently before combining. The real,
final result — 57.78% test / 57.36% forget-client accuracy — is a
genuine middle ground: much less collateral damage than GA alone
(−2.24 vs. −4.61 points) while still meaningfully forgetting (−9.74
points vs. sequential KD's −0.15). Not a strict win on both axes over
GA alone, but a real, working balance point — see
`phases/phase-07-unlearning-engine/PHASE.md` for the full bug/fix story
and `docs/results.md` for the complete writeup. A λ_forget/λ_kd sweep
(not yet done) would trace the full tradeoff curve rather than one
point on it — natural next step if more GPU time becomes available.

```
Data pipeline      ██████████ 100%
FL baseline        ██████████ 100%
Evaluation utils   ████░░░░░░  40%   (accuracy/cost/comparison done; forgetting/MIA blocked on unlearning)
Full retraining    ██████████ 100%
Gradient ascent    ██████████ 100%
Knowledge distill. ██████████ 100%
Unlearning engine  ██████████ 100%
MIA                ░░░░░░░░░░   0%
Experiments/report ░░░░░░░░░░   0%
```

## Phase checklist

- [x] Phase 00 — Repo & Skeleton
- [x] Phase 01 — Data Pipeline
- [x] Phase 02 — Model + FedAvg
- [x] Phase 03 — Initial FL Experiment (real result: CIFAR-100+ResNet-18, M_old = 60.02% test acc.)
- [x] Phase 04 — Full Retraining Baseline (real result: M_retrain = 70.89% test acc., 65.30% forget-client acc., via FedProx — see history above)
- [x] Phase 05 — Gradient Ascent (real result: M_unlearn = 55.41% test acc., 52.31% forget-client acc. — see interpretation above)
- [x] Phase 06 — Knowledge Distillation (real result: sequential KD repairs accuracy but also undoes forgetting — see finding above)
- [x] Phase 07 — Unlearning Engine (real result: joint engine = 57.78% test acc., 57.36% forget-client acc. — see result above)
- [ ] Phase 08 — Evaluation Framework (partial — see above)
- [ ] Phase 09 — MIA
- [ ] Phase 10 — Controlled Experiments
- [ ] Phase 11 — Plots + Reports
- [ ] Phase 12 — Final Report / Viva

## Next action

Start Phase 09 (MIA) in `phases/phase-09-mia/PHASE.md`, or consider a
λ_forget/λ_kd sweep on Phase 07's engine first if more GPU time is
available — either strengthens the thesis, MIA adds a rigorous privacy
metric beyond accuracy, the sweep shows the full tradeoff curve rather
than one point on it.
