# Phase 07 — Unlearning Engine

**Status:** Done — real result obtained (57.78% test acc., 57.36% forget-client acc.)
**Depends on:** Phase 05, Phase 06

## Goal

Combine Gradient Ascent + Knowledge Distillation into the end-to-end unlearning pipeline that produces `M_unlearn`.

## Tasks

- [x] Implement `src/unlearning/engine.py::UnlearningEngine.run`
- [x] ~~Calls `GradientAscentUnlearner` then `KnowledgeDistiller` per blueprint Section 9~~ — see Handoff notes: implemented as a **joint** loss instead, per `docs/methodology.md`
- [x] Save `M_unlearn` checkpoint + metrics under `artifacts/experiments/unlearning_ga_kd/`
- [x] Use `src/unlearning/forget_client.py` for client selection (already implemented)

## Definition of done

`UnlearningEngine.run()` returns an `UnlearningResult` with a checkpointed `M_unlearn` and recorded metrics.

## Handoff notes

**Deviated from this file's original task wording ("calls GA then KD")
on purpose, with evidence.** Phase 06 empirically showed that running
GA then KD *sequentially* undoes almost all of GA's forgetting: KD's
only signal is "match the teacher on remaining-client data," and under
non-IID class overlap between clients, that ends up restoring
forget-client behavior too (52.31% → 66.95% forget-client accuracy —
back near M_old's 67.10%). No "blueprint" document with numbered
sections exists anywhere in this repo's git history (checked `git log
--all`) — but `docs/methodology.md` does specify the actual intended
objective explicitly:

    L_total = lambda_forget * L_forget + lambda_kd * L_KD

a single **combined** loss, not two sequential stages. Implemented
`UnlearningEngine.run()` accordingly: one training loop, each step
pairs a batch from the forget client with a batch from the remaining
clients, computes both losses against the same student, combines them
with `lambda_forget`/`lambda_kd`, and does one backward pass. The
student starts as a copy of `M_old` (teacher); `M_old` itself is used
read-only and never mutated. Gradient clipping (`max_grad_norm`,
reused from the `gradient_ascent` config section) prevents the
unbounded forgetting term from diverging, same issue as Phase 05.

Three unit tests in `tests/test_unlearning_engine.py`. The important
one, `test_joint_engine_forgets_without_collapsing_remaining_accuracy`,
specifically checks the property Phase 06 showed sequential
application fails at: forget-loss increases AND remaining-client
accuracy doesn't collapse, in the *same* run. All 13 repo tests pass.

Added `experiments/run_unlearning_engine.py`: loads `M_old`, builds
forget/remaining loaders from the real partition, runs the joint
engine, evaluates test/forget-client accuracy before/after. Reuses
`knowledge_distillation`'s config section for `lr`/`epochs`/
`temperature`/`lambda_forget`/`lambda_kd` (it already carried
`lambda_forget` unused in Phase 06, clearly anticipating this engine)
and `gradient_ascent`'s `batch_size`/`max_grad_norm` for the forget
side.

**First real GPU result (before a bug fix, kept here for the record):**

| Model | Test acc. | Forget-client acc. |
|---|---:|---:|
| M_old | 60.02% | 67.10% |
| GA only (Phase 05) | 55.41% | 52.31% |
| GA+KD sequential (Phase 06) | 59.99% | 66.95% |
| GA+KD joint (first attempt) | 50.00% | 38.13% |

This was worse than plain GA on *both* axes — defeating the purpose of
adding KD at all. `training_history` showed why: `kd_loss` increased
every epoch (0.12 → 0.45) instead of decreasing, meaning the
preservation objective never converged. Root cause: gradient clipping
(`max_grad_norm`) was applied to the *combined* `lambda_forget *
L_forget + lambda_kd * L_KD` loss's gradient as one sum. Since the
unbounded ascent term's raw gradient magnitude is much larger than
KD's bounded KL-divergence term, clipping the sum still left the
update direction dominated by the forgetting term almost entirely,
regardless of the nominal 1:1 lambda weighting — clipping only rescales
magnitude, it doesn't rebalance which term controls direction.

**Fix:** backprop each loss separately (`torch.autograd.grad` per
objective) and clip each gradient independently (new
`_clip_grads_by_norm` helper, since `torch.nn.utils.clip_grad_norm_`
only operates on parameters' populated `.grad`, not arbitrary gradient
tensor lists) *before* combining them with `lambda_forget`/`lambda_kd`
into the actual parameter update. Verified on a realistically
under-fit (not overfit) synthetic model: with the exact same
`lr=0.001, epochs=5, lambda_kd=1.0` config values that produced the bad
result above, the fixed version now drops remaining accuracy only
moderately (67.2% → 57.8%) while forgetting *more* (62.5% → 43.8%) —
forgetting outpacing collateral damage, the correct direction, unlike
before. All 13 repo tests still pass.

**Confirmed on real CIFAR-100 (GPU, Colab), same hyperparameters:**

- `M_unlearn` (joint, fixed) overall test accuracy: **57.78%**
- `M_unlearn` (joint, fixed) forget-client accuracy: **57.36%**

`training_history` in the committed `metrics.json` confirms the fix
worked as diagnosed: `kd_loss` stayed nearly flat across epochs
(0.105 → 0.123), versus nearly quadrupling (0.12 → 0.45) in the buggy
version.

This is a genuine middle ground, not a strict win over GA alone on both
axes — it trades some forgetting strength for much less collateral
damage: only −2.24 points of overall accuracy loss (vs. GA's −4.61),
while still meaningfully forgetting (−9.74 points on the forget client,
vs. sequential KD's essentially-zero −0.15). With equal λ weights, this
is one point on a real precision-vs-cost tradeoff curve. A proper
λ_forget/λ_kd sweep (not yet done — would need more GPU time than one
Colab session) would trace that whole curve rather than one point on
it, and is the natural next step if more compute becomes available
(the professor's offered GPU access would be well spent here).

Artifacts: `artifacts/experiments/unlearning_ga_kd/engine/`
(`m_unlearn_final.pt`, `metrics.json`, `config.yaml`). Full writeup
with chart: `docs/results.md`.
