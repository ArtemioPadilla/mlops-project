# Complete Pipeline Workflow

End-to-end workflow from raw data to trained models.

## Overview

```bash
make pipeline  # Runs: preprocess + train
```

## Step-by-Step

### 1. Preprocessing

```bash
make preprocess
```

**What happens**:
1. Load `data/raw/online_news_modified.csv`
2. Clean data (URL validation, numeric conversion, business rules)
3. Engineer features (classify binary/non-binary)
4. Split 70/15/15 (train/val/test)
5. Handle correlation (train-only)
6. Save to `data/processed/`

**Outputs**:
- `X_train.csv`, `X_val.csv`, `X_test.csv`
- `y_train.csv`, `y_val.csv`, `y_test.csv`
- `metadata.json`

### 2. Model Training

```bash
make train
```

**What happens**:
1. Load processed data
2. For each model in `config/models.yaml`:
   - Build sklearn Pipeline
   - Train model
   - Evaluate on train/val/test
   - Log to MLflow
3. Select best model
4. Save to `models/`

**Outputs**:
- MLflow experiments in `mlflow/dev/`
- Best model in `models/`

### 3. View Results

```bash
make mlflow-ui
```

Open http://localhost:5001 to view experiments.

## Next Steps

- [Development Workflow](development.md)
- [Experiment Workflow](experiments.md)
