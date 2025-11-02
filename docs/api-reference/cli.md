# CLI API Reference

Complete API reference for command-line interfaces. All CLI modules are auto-generated from source code docstrings.

## Preprocessing CLI

::: mlops_online_news_popularity.cli.preprocess_cli
    options:
      show_root_heading: true
      show_source: true
      members_order: source
      show_signature_annotations: true

## Training CLI

::: mlops_online_news_popularity.cli.train_cli
    options:
      show_root_heading: true
      show_source: true
      members_order: source
      show_signature_annotations: true

---

## Usage Examples

### Preprocessing Pipeline

Run the complete preprocessing pipeline:

```bash
# Basic usage
python -m mlops_online_news_popularity.cli.preprocess_cli \
  --input data/raw/online_news_modified.csv \
  --output-dir data/processed

# With custom correlation threshold
python -m mlops_online_news_popularity.cli.preprocess_cli \
  --input data/raw/online_news_modified.csv \
  --output-dir data/processed \
  --corr-threshold 0.85
```

### Model Training

Train a single model:

```bash
# Train Ridge regression
python -m mlops_online_news_popularity.cli.train_cli train-single --model ridge

# Train Random Forest
python -m mlops_online_news_popularity.cli.train_cli train-single --model randomforest
```

Train and compare multiple models:

```bash
# Compare all models from config file
python -m mlops_online_news_popularity.cli.train_cli train-compare config/models.yaml
```

See the [CLI Commands Guide](../cli/commands.md) for more examples and detailed usage.
