# Quick Start Guide

Get up and running with the MLOps pipeline in 5 minutes.

## Prerequisites

- Python 3.10 installed
- Git installed
- Virtual environment tool (venv, conda, or virtualenvwrapper)

## Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/artemiopadilla/mlops-project.git
cd mlops-project

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in editable mode (REQUIRED!)
pip install -e .

# Install dependencies
pip install -r requirements.txt
```

!!! warning "Important"
    You MUST run `pip install -e .` to install the package in editable mode. This allows Python to import `mlops_online_news_popularity` modules correctly.

---

## Step 2: Run the Complete Pipeline

The fastest way to see results is to run the complete pipeline:

```bash
make pipeline
```

This single command will:

1. ✅ Preprocess raw data
2. ✅ Create train/val/test splits (70/15/15)
3. ✅ Train multiple models (Ridge, RandomForest, KNeighbors)
4. ✅ Track experiments in MLflow
5. ✅ Save the best model to `models/`

**Expected output**:
```
PREPROCESSING PIPELINE - CLI
==============================================================
Input: data/raw/online_news_modified.csv
...
PREPROCESSING COMPLETE
==============================================================

TRAINING MODELS
==============================================================
Training Ridge...
Training RandomForest...
Training KNeighbors...

Best Model: RandomForest
Validation RMSE: 0.8234
Saved to: models/randomforest_best_20241102_001256.pkl
```

---

## Step 3: View Results in MLflow UI

Start the MLflow user interface to explore experiments:

```bash
make mlflow-ui
```

Then open http://localhost:5001 in your browser.

**What you'll see**:

- All model runs with metrics (RMSE, MAE, R²)
- Comparison charts across train/val/test sets
- Model artifacts (saved pipelines)
- Parameters and tags

---

## Alternative: Step-by-Step Execution

If you prefer to run each step individually:

### Preprocess Data Only

```bash
make preprocess
```

This creates:
```
data/processed/
├── X_train.csv
├── X_val.csv
├── X_test.csv
├── y_train.csv
├── y_val.csv
├── y_test.csv
└── metadata.json
```

### Train Models Only

```bash
make train
```

This trains all models defined in `config/models.yaml`.

### Train a Single Model (Fast Testing)

```bash
make train-single

# Or specify a model
python -m mlops_online_news_popularity.cli.train_cli train-single --model ridge
```

---

## Step 4: Explore Data Profiling Reports

Profiling reports are generated during preprocessing and saved to `docs/assets/html/`:

- **Raw Data Report**: `01_raw_data_report.html`
- **Cleaned Data Report**: `02_cleaned_data_report.html`
- **Train Set Report**: `03_train_set_report.html`
- **Test Set Report**: `04_test_set_report.html`

Open any report in your browser to see:

- Dataset statistics
- Feature distributions
- Correlations
- Missing values analysis
- Data quality warnings

---

## Verify Your Setup

Test that everything works:

```bash
# 1. Check imports work
python -c "from mlops_online_news_popularity import preprocessing, modeling, config; print('✓ Imports OK')"

# 2. Check data exists
ls data/raw/online_news_modified.csv

# 3. Check MLflow database
ls mlflow/dev/mlflow.db
```

---

## Common Issues

### ModuleNotFoundError: No module named 'mlops_online_news_popularity'

**Solution**: Run `pip install -e .` from the project root.

### FileNotFoundError: data/raw/online_news_modified.csv

**Solution**: Ensure you have the raw data file. If using DVC:
```bash
dvc pull
```

### MLflow UI shows no experiments

**Solution**: Run `make train` first to create experiments.

---

## Next Steps

Now that you have a working setup, explore:

- [Complete Pipeline Workflow](complete-pipeline.md) - Understand each step in detail
- [Configuration Guide](../configuration/models-yaml.md) - Customize models and hyperparameters
- [Preprocessing Overview](../preprocessing/overview.md) - Learn about data preprocessing
- [Model Training](../modeling/model-trainer.md) - Deep dive into model training

---

## Quick Command Reference

| Command | What it does |
|---------|--------------|
| `make pipeline` | Run full pipeline (preprocess + train) |
| `make preprocess` | Preprocess data only |
| `make train` | Train all models from config |
| `make train-single` | Train a single model quickly |
| `make mlflow-ui` | Start MLflow UI on port 5001 |
| `make lint` | Check code quality |
| `make test` | Run pytest tests |
| `make docs-serve` | Serve documentation locally |

---

!!! success "You're Ready!"
    You now have a fully functional MLOps pipeline. Start experimenting with different models and configurations!
