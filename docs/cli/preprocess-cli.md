# Preprocessing CLI

Command-line interface for data preprocessing.

## Usage

```bash
python -m mlops_online_news_popularity.cli.preprocess_cli [OPTIONS]
```

## Options

- `--input`, `-i`: Path to raw CSV file (default: `data/raw/online_news_modified.csv`)
- `--output-dir`, `-o`: Directory to save processed data (default: `data/processed/`)
- `--target`, `-t`: Name of target column (default: "shares")
- `--corr-threshold`: Correlation threshold for feature removal (default: 0.9)

## Examples

```bash
# Default settings
python -m mlops_online_news_popularity.cli.preprocess_cli

# Custom input and output
python -m mlops_online_news_popularity.cli.preprocess_cli \
  --input data/raw/my_data.csv \
  --output-dir data/my_processed \
  --corr-threshold 0.85
```

## Output Files

- `X_train.csv`, `X_val.csv`, `X_test.csv`
- `y_train.csv`, `y_val.csv`, `y_test.csv`
- `metadata.json`
