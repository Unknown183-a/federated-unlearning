# Roadmap

Three-month project, tracked in phases. Only Phases 1-2 are complete;
everything after that is scaffolded (interfaces exist) but not
implemented, per the project's honesty requirement — no results are
fabricated ahead of running real experiments.

| Phase | Deliverable | Status |
|---|---|---|
| 1. Proposal | Problem statement, objectives, initial architecture | ✅ Completed |
| 2. FL Baseline | MNIST + 5-client sim + FedAvg + 50 rounds x 10 local epochs | ✅ Completed (`experiments/run_initial_fl.py`) |
| 3. Full Retraining | Forget-client removal + retrain-from-scratch baseline (`M_retrain`) | ⏳ Next |
| 4. Gradient Ascent | `src/unlearning/gradient_ascent.py` implementation | ⏳ Planned |
| 5. Knowledge Distillation | `src/unlearning/knowledge_distillation.py` implementation | ⏳ Planned |
| 6. Evaluation | Accuracy/forgetting/MIA/cost comparison, `M_unlearn` vs `M_retrain` | ⏳ Planned |
| 7. Controlled experiments | Client-count, IID/Non-IID, hyperparameter grids (small, controlled) | ⏳ Planned |
| 8. Final report | Plots, tables, viva prep | ⏳ Planned |

## Current experiment label

**Initial Federated Learning Pipeline Validation** — 5 clients, 50
communication rounds, 10 local epochs/round, MNIST, FedAvg. This
setup is for pipeline validation, not the final experimental
configuration, and its result (once run) should be read from
`artifacts/experiments/initial_fl/metrics.csv`, never fabricated.

## Talking points for the current evaluation

> After the project proposal, I focused on understanding the technical
> foundations and building the initial Federated Learning pipeline. I
> prepared MNIST data and distributed it among five simulated clients.
> As an initial pipeline-validation experiment, I trained the model for
> 50 communication rounds, with each client performing 10 local epochs
> per round. This confirmed that the basic FL training and aggregation
> pipeline works. The next phase is to create the full-retraining
> baseline and then implement Gradient Ascent-based unlearning followed
> by Knowledge Distillation.

## Future extensions (optional, not required completed work)

More clients, Non-IID partitions, other datasets/architectures,
multiple/repeated forget requests, stronger MIA, differential privacy,
secure aggregation, communication-efficient unlearning, larger-scale
experiments.
