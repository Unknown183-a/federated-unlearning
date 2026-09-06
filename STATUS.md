# Status

Source of truth for where the project actually stands. Kept in sync
with `README.md`'s Current Phase checklist above the fold — update
both when a phase finishes.

## Current phase

**Phase 07 — Unlearning Engine** (not started)

Phases 00-06 are complete with real results, all against the real
CIFAR-100 + ResNet-18 setup. M_old was imported from a pre-refactor
checkpoint; M_retrain was retrained via FedProx on GPU (Colab); M_unlearn
(Gradient Ascent) was produced by running 5 epochs of ascent on the
forget client's data starting from M_old; Knowledge Distillation was
run standalone (sequentially, after GA) to test whether it repairs GA's
collateral damage. Phase 08 (Evaluation Framework) is partially done
ahead of schedule — accuracy/cost/comparison utilities exist since they
only depend on Phase 03, not on the unlearning engine.

**Results** (CIFAR-100, 5 clients, non-IID Dirichlet α=0.5, forget client = 0):

| Model                                  | Overall test acc. | Forget-client acc. | Cost |
|------------------------------------------|-------------------:|---------------------:|------|
| M_old                                     | 60.02%             | 67.10%                | 20 rounds × 2 local epochs |
| M_retrain (FedProx, gold standard)        | 70.89%             | 65.30%                | 50 rounds × 10 local epochs, μ=0.01 |
| M_unlearn (Gradient Ascent only)          | 55.41%             | 52.31%                | 5 epochs on forget client only |
| M_unlearn + sequential KD                 | 59.99%             | 66.95%                | + 5 epochs KD on remaining clients |

**Interpretation:** Gradient Ascent forgot client 0's data faster and
far cheaper than full retraining, but less precisely: forget-client
accuracy dropped 15.0 points (67.10% → 52.31%) while overall test
accuracy also dropped 4.6 points (60.02% → 55.41%) — collateral damage
to unrelated knowledge, a known weakness of plain gradient ascent. By
contrast, M_retrain improved on *both* axes relative to M_old, since
it's a from-scratch fit rather than a targeted edit.

**Critical finding from Phase 06:** applying KD *sequentially* after GA
repaired overall accuracy almost perfectly (55.41% → 59.99%, ~M_old
level) — but also undid nearly all of the forgetting (52.31% → 66.95%,
back near M_old's 67.10%). Pure KD's only signal is "match the teacher
on remaining-client data," with no instruction to preserve GA's
forgetting, and non-IID class overlap between clients means matching
M_old on remaining data appears to restore behavior on overlapping
forget-client classes too. **This means Phase 07's `UnlearningEngine`
cannot just call `GradientAscentUnlearner` then `KnowledgeDistiller` in
sequence — it must interleave both objectives within the same training
loop** (alternating or jointly-weighted GA+KD steps, using
`lambda_forget`/`lambda_kd` from `configs/unlearning.yaml` to balance
them), or the KD half will erase the GA half's entire effect. This is
the central design question Phase 07 needs to get right.

```
Data pipeline      ██████████ 100%
FL baseline        ██████████ 100%
Evaluation utils   ████░░░░░░  40%   (accuracy/cost/comparison done; forgetting/MIA blocked on unlearning)
Full retraining    ██████████ 100%
Gradient ascent    ██████████ 100%
Knowledge distill. ██████████ 100%
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
- [x] Phase 05 — Gradient Ascent (real result: M_unlearn = 55.41% test acc., 52.31% forget-client acc. — see interpretation above)
- [x] Phase 06 — Knowledge Distillation (real result: sequential KD repairs accuracy but also undoes forgetting — see critical finding above)
- [ ] Phase 07 — Unlearning Engine
- [ ] Phase 08 — Evaluation Framework (partial — see above)
- [ ] Phase 09 — MIA
- [ ] Phase 10 — Controlled Experiments
- [ ] Phase 11 — Plots + Reports
- [ ] Phase 12 — Final Report / Viva

## Next action

Start Phase 07 (Unlearning Engine) in
`phases/phase-07-unlearning-engine/PHASE.md`. Given the critical
finding above, the engine MUST interleave Gradient Ascent and
Knowledge Distillation steps rather than sequencing them — verify the
PHASE.md spec agrees before implementing, and if it's ambiguous on
this point, interleave anyway based on this evidence.
