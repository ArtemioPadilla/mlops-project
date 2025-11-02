# Training CLI

Command-line interface for model training.

## Commands

### train-compare

Train and compare multiple models from YAML config.

```bash
python -m mlops_online_news_popularity.cli.train_cli train-compare CONFIG_PATH
```

**Example**:
```bash
python -m mlops_online_news_popularity.cli.train_cli train-compare config/models.yaml
```

### train-single

Train a single model for quick testing.

```bash
python -m mlops_online_news_popularity.cli.train_cli train-single [OPTIONS]
```

**Options**:
- `--model`: Model type (default: "ridge")
  - Choices: ridge, randomforest, kneighbors, xgboost

**Examples**:
```bash
# Default (Ridge)
python -m mlops_online_news_popularity.cli.train_cli train-single

# RandomForest
python -m mlops_online_news_popularity.cli.train_cli train-single --model randomforest
```
