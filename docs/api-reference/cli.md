# CLI API Reference

Command-line interface reference.

## preprocess_cli

**Module**: `mlops_online_news_popularity.cli.preprocess_cli`

### Command

```bash
python -m mlops_online_news_popularity.cli.preprocess_cli [OPTIONS]
```

### Options

- `--input`, `-i TEXT`: Path to raw CSV
- `--output-dir`, `-o TEXT`: Output directory
- `--target`, `-t TEXT`: Target column name
- `--corr-threshold FLOAT`: Correlation threshold

## train_cli

**Module**: `mlops_online_news_popularity.cli.train_cli`

### Commands

#### train-compare

```bash
python -m mlops_online_news_popularity.cli.train_cli train-compare CONFIG_PATH
```

#### train-single

```bash
python -m mlops_online_news_popularity.cli.train_cli train-single [OPTIONS]
```

Options:
- `--model TEXT`: Model type (ridge, randomforest, kneighbors, xgboost)
