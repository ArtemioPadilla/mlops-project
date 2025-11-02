# CLI Commands Overview

Command-line interfaces for the MLOps pipeline.

## Available Commands

### Preprocessing

```bash
python -m mlops_online_news_popularity.cli.preprocess_cli [OPTIONS]
```

See [Preprocessing CLI](preprocess-cli.md).

### Training

```bash
python -m mlops_online_news_popularity.cli.train_cli COMMAND [OPTIONS]
```

Commands:
- `train-compare`: Train and compare multiple models
- `train-single`: Train a single model

See [Training CLI](train-cli.md).

## Makefile Shortcuts

| Command | Equivalent |
|---------|------------|
| `make preprocess` | `python -m ... preprocess_cli` |
| `make train` | `python -m ... train_cli train-compare config/models.yaml` |
| `make train-single` | `python -m ... train_cli train-single` |
| `make pipeline` | `make preprocess && make train` |
