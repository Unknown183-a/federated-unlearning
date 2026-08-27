# Federated Unlearning via Gradient Ascent + Knowledge Distillation

Course-project repository. Federated Learning trains a shared model
across clients without pooling raw data; this project studies how to
efficiently make a trained federated model "forget" one client's
contribution, without paying the cost of retraining from scratch.

See `docs/architecture.md` for the full bird's-eye system diagram and
`docs/methodology.md` / `docs/roadmap.md` for the approach and status.

## Status

- ✅ **Working now:** MNIST data pipeline, IID/Non-IID client
  partitioning, FedAvg federated training, checkpointing, metrics
  logging — i.e. everything needed to produce `M_old`.
- ⏳ **Scaffolded, not yet implemented:** full-retraining baseline,
  Gradient Ascent unlearning, Knowledge Distillation, MIA evaluation.
  These raise `NotImplementedError` with a pointer to the roadmap
  phase that will fill them in — see `docs/roadmap.md`.

No experimental results are fabricated anywhere in this repo. Numbers
only ever come from `artifacts/experiments/*/metrics.csv` after an
actual run.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run the initial FL experiment

```bash
python experiments/run_initial_fl.py --config configs/initial_fl.yaml
```

This trains the pipeline-validation configuration (5 clients, 50
rounds, 10 local epochs/round) and writes to
`artifacts/experiments/initial_fl/`:

```text
config.yaml            # exact config used
metrics.csv            # per-round metrics
training.log           # training log
partition_metadata.json
checkpoints/round_*.pt
```

## Tests

```bash
pytest tests/
```

## Repository layout

```text
configs/            # experiment configuration (YAML) — no hard-coded params in code
src/data/           # MNIST loading + IID/Non-IID partitioning
src/models/         # model architectures (small CNN)
src/federated/      # client, server, FedAvg, round tracking
src/unlearning/     # forget-client selection, Gradient Ascent, KD, engine (partly scaffold)
src/baselines/      # full-retraining reference (scaffold)
src/evaluation/     # accuracy, forgetting, MIA, cost, comparison table
src/utils/          # config loading, logging, checkpointing, seeding
experiments/        # runnable experiment scripts
artifacts/          # experiment outputs (configs, metrics, logs, checkpoints)
docs/               # architecture, methodology, roadmap
tests/              # unit tests
```

## Engineering principles

Separation of concerns across the module boundaries above; every
result reproducible from a saved config; no hard-coded experiment
parameters; every experiment checkpoints its model(s); fair
comparisons use the same architecture and evaluation data; results are
never fabricated.
