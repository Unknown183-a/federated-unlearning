# Status

Source of truth for where the project actually stands. Kept in sync
with `README.md`'s Current Phase checklist above the fold — update
both when a phase finishes.

## Current phase

**Phase 06 — Knowledge Distillation** (not started)

Phases 00-05 are complete with real results, all against the real
CIFAR-100 + ResNet-18 setup. M_old was imported from a pre-refactor
checkpoint; M_retrain was retrained via FedProx on GPU (Colab); M_unlearn
(Gradient Ascent) was produced by running 5 epochs of ascent on the
forget client's data starting from M_old, also on GPU. Phase 08
(Evaluation Framework) is partially done ahead of schedule —
accuracy/cost/comparison utilities exist since they only depend on
Phase 03, not on the unlearning engine.

**Results** (CIFAR-100, 5 clients, non-IID Dirichlet α=0.5, forget client = 0):

| Model                          | Overall test acc. | Forget-client acc. | Cost |
|---------------------------------|-------------------:|---------------------:|------|
| M_old                           | 60.02%             | 67.10%                | 20 rounds × 2 local epochs |
| M_retrain (FedProx, gold standard) | 70.89%          | 65.30%                | 50 rounds × 10 local epochs, μ=0.01 |
| M_unlearn (Gradient Ascent)      | 55.41%             | 52.07%                | 5 epochs on forget client only |

**Interpretation:** Gradient Ascent forgot client 0's data faster and
far cheaper than full retraining, but less precisely: forget-client
accuracy dropped 15.0 points (67.10% → 52.07%) while overall test
accuracy also dropped 4.6 points (60.02% → 55.41%) — some collateral
damage to unrelated knowledge, a known weakness of plain gradient
ascent (it damages whatever activates similarly to the forget class,
not just the forget class itself). By contrast, M_retrain never saw
this tradeoff: it improved on *both* axes relative to M_old, since it's
a from-scratch fit rather than a targeted edit. This cost-vs-precision
tradeoff (GA: cheap but imprecise; retraining: precise but expensive)
is exactly what Phase 06's Knowledge Distillation approach and Phase 07's
combined engine are meant to improve on.

```
Data pipeline      ██████████ 100%
FL baseline        ██████████ 100%
Evaluation utils   ████░░░░░░  40%   (accuracy/cost/comparison done; forgetting/MIA blocked on unlearning)
Full retraining    ██████████ 100%
Gradient ascent    ██████████ 100%
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
- [x] Phase 05 — Gradient Ascent (real result: M_unlearn = 55.41% test acc., 52.07% forget-client acc. — see interpretation above)
- [ ] Phase 06 — Knowledge Distillation
- [ ] Phase 07 — Unlearning Engine
- [ ] Phase 08 — Evaluation Framework (partial — see above)
- [ ] Phase 09 — MIA
- [ ] Phase 10 — Controlled Experiments
- [ ] Phase 11 — Plots + Reports
- [ ] Phase 12 — Final Report / Viva

## Next action

Start Phase 06 (Knowledge Distillation) in
`phases/phase-06-knowledge-distillation/PHASE.md`, using
`artifacts/experiments/cifar100_fl/imported_original_model.pt` as M_old,
`artifacts/experiments/full_retraining/m_retrain_final.pt` as M_retrain,
and `artifacts/experiments/unlearning_ga_kd/gradient_ascent/m_unlearn_final.pt`
as the Gradient Ascent result to compare KD-based unlearning against.
