# Reproducibility Guide

This document explains how the MLOps Online News Popularity project ensures reproducible results across different environments and how to validate reproducibility.

## Table of Contents

- [Overview](#overview)
- [Reproducibility Guarantees](#reproducibility-guarantees)
- [Quick Validation](#quick-validation)
- [Detailed Validation Process](#detailed-validation-process)
- [Expected Results](#expected-results)
- [Artifacts Versioning](#artifacts-versioning)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## Overview

Reproducibility is a cornerstone of MLOps, enabling:
- **Auditability**: Trace model behavior to exact code/data versions
- **Debugging**: Isolate issues by reproducing exact conditions
- **Collaboration**: Ensure team members get identical results
- **Compliance**: Meet regulatory requirements for model governance

This project achieves reproducibility through multiple mechanisms:

---

## Reproducibility Guarantees

### 1. Python Version Control

**File**: `.python-version`
```
3.10
```

**Benefits**:
- `pyenv` automatically activates Python 3.10
- IDEs detect correct interpreter
- CI/CD uses same version

**Enforcement**:
- `pyproject.toml`: `requires-python = "~=3.10.0"`
- GitHub Actions: `python-version: '3.10'`

**Verification**:
```bash
python --version
# Expected: Python 3.10.x
```

---

### 2. Dependency Pinning

**File**: `requirements.txt`

All 62+ dependencies use **exact version pinning** with `==`:

```txt
numpy==2.2.6
pandas==2.2.3
scikit-learn==1.5.2
mlflow==3.6.0
fastapi==0.121.2
pytest==9.0.1
black==25.11.0
# ... all dependencies pinned
```

**Why exact pinning?**
- Prevents breaking changes from library updates
- Ensures identical package versions across environments
- Reproducible builds months/years later

**Verification**:
```bash
pip freeze | grep -E "numpy|pandas|scikit-learn"
# Should match requirements.txt exactly
```

---

### 3. Random Seed Configuration

**Fixed seed**: `random_state=42` throughout the codebase

**Locations**:
- `preprocessing/data_processor.py:43` - Train/test splits
- `cli/train_cli.py:33-85` - All model training
- `modeling/compare.py:67` - Automatic seed injection
- `tests/test_serving/conftest.py:109` - Test fixtures

**Example**:
```python
# Data splitting
train_test_split(X, y, random_state=42, shuffle=True)

# Model training
RandomForestRegressor(random_state=42)
Ridge(random_state=42)
```

**Verification**:
```bash
# Run preprocessing twice
make preprocess
md5sum data/processed/X_train.csv  # Note hash

# Clean and rerun
rm -rf data/processed/*
make preprocess
md5sum data/processed/X_train.csv  # Hash should match
```

---

### 4. Data Versioning (DVC)

**Files**:
- `data.dvc` - Tracks raw data
- `models.dvc` - Tracks trained models
- `.dvc/config` - Remote storage configuration

**Usage**:
```bash
# Pull specific version
dvc checkout data.dvc@v1.0.0

# Verify data integrity
dvc status
```

---

### 5. Experiment Tracking (MLflow)

**Location**: `mlflow_artifacts/dev/mlflow.db`

**Tracked artifacts**:
- Model binaries (pickle files)
- Hyperparameters
- Metrics (RMSE, MAE, R²)
- Feature names and preprocessing steps
- Git commit SHA

**Usage**:
```bash
# View experiments
make mlflow-ui
# Open: http://localhost:5001

# Load specific run
mlflow.sklearn.load_model("runs:/<run_id>/model_pipeline")
```

---

## Quick Validation

### CI vs Local Testing

The project uses **different datasets** for CI and local testing:

| Environment | Dataset | Size | Purpose |
|-------------|---------|------|---------|
| **GitHub Actions CI** | `data/sample/online_news_sample.csv` | 2,000 rows | Fast validation in CI pipeline |
| **Local Development** | `data/raw/online_news_modified.csv` | ~40,000 rows | Full validation with complete dataset |

Both validate reproducibility correctly, but local testing with the full dataset provides more comprehensive validation.

**Why use sample data in CI?**
- ✅ Faster CI runs (~5-10 min vs ~15-20 min)
- ✅ Sample data can be committed to git (no DVC setup needed)
- ✅ Still validates reproducibility guarantees
- ⚠️ Metrics will differ from local runs (different data size)

### Using Makefile Targets (Recommended)

The project provides multiple Makefile targets for testing reproducibility:

#### 1. Strict Mode (Production)

```bash
make test-reproducibility
```

**Features:**
- Validates Python 3.10 is active
- Runs complete validation (splits + metrics + predictions)
- Recommended for CI/CD and official validation
- Fails if Python version != 3.10

**Use when:**
- Running in CI/CD pipeline
- Official validation before release
- Have Python 3.10 environment active

---

#### 2. Development Mode

```bash
make test-reproducibility-dev
```

**Features:**
- Allows any Python 3.x version
- Same validation as strict mode
- ⚠️ Results may differ from production if not using Python 3.10

**Use when:**
- Local development with Python 3.14+
- Quick iteration during development
- Python 3.10 not available locally

---

#### 3. Docker Mode

```bash
make test-reproducibility-docker
```

**Features:**
- Runs test in Docker container with Python 3.10
- Guaranteed consistent environment
- No local Python 3.10 required
- Slower but 100% reliable

**Use when:**
- Python 3.10 not installed locally
- Need absolute guarantee of Python 3.10
- Running on different OS than production

---

#### 4. Quick Test

```bash
make test-reproducibility-quick
```

**Features:**
- Only compares Test RMSE metrics
- ~2x faster than full test
- Skips file comparisons

**Use when:**
- Quick sanity check during development
- Testing after minor code changes
- Want fast feedback loop

---

#### 5. Check Python Version

```bash
make check-python
```

**Features:**
- Only validates Python version
- Shows clear error messages if version mismatch
- Suggests solutions

**Use when:**
- Verifying environment setup
- Before running reproducibility tests
- Debugging version issues

---

### Using Custom Datasets

You can test reproducibility with any dataset using the `REPRO_DATA_PATH` environment variable:

```bash
# Test with sample data (fast, for CI)
REPRO_DATA_PATH=data/sample/online_news_sample.csv bash scripts/test_reproducibility.sh

# Test with full data (default)
bash scripts/test_reproducibility.sh

# Test with custom dataset
REPRO_DATA_PATH=path/to/your/dataset.csv bash scripts/test_reproducibility.sh
```

### Regenerating Sample Dataset

If you need to regenerate the sample dataset (e.g., after updating the full dataset):

```bash
python scripts/create_sample_data.py
```

This creates `data/sample/online_news_sample.csv` with 2,000 rows sampled from the full dataset.

---

### Automated Test (Direct Script Execution)

For advanced users or custom workflows:

```bash
bash scripts/test_reproducibility.sh
```

Expected output:
```
===========================================
REPRODUCIBILITY TEST
===========================================
Step 1: Verifying Python version...
✅ Python 3.10 detected

Step 2: Verifying package installation...
✅ Package installed

Step 3: Setting up test directories...
✅ Test directories created

Step 4: Running pipeline (Run 1)...
✅ Run 1 completed

Step 5: Running pipeline (Run 2)...
✅ Run 2 completed

Step 6: Comparing results...
✅ Data splits match
✅ Metrics match exactly
✅ Model predictions identical

===========================================
✅ REPRODUCIBILITY TEST PASSED
===========================================
```

---

### Comparison Matrix

| Feature | `test-reproducibility` | `test-reproducibility-dev` | `test-reproducibility-docker` | `test-reproducibility-quick` |
|---------|------------------------|----------------------------|-------------------------------|------------------------------|
| **Python 3.10 required** | ✅ Yes | ❌ No | ✅ Yes (automatic) | ⚠️ Depends on environment |
| **Speed** | Medium (~5 min) | Medium (~5 min) | Slow (~10 min) | Fast (~2 min) |
| **Reliability** | High | Medium | Highest | Medium |
| **Use case** | Production | Development | Any environment | Quick check |
| **Fails on version mismatch** | ✅ Yes | ❌ No | N/A | ❌ No |

---

## Detailed Validation Process

### Step 1: Clean Environment Setup

```bash
# Remove existing environment
rm -rf venv

# Create fresh Python 3.10 virtual environment
python3.10 -m venv venv_clean
source venv_clean/bin/activate  # Windows: venv_clean\Scripts\activate

# Verify Python version
python --version  # Must be 3.10.x
```

### Step 2: Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install exact versions from requirements.txt
pip install -r requirements.txt

# Install package in editable mode
pip install -e .

# Verify installations
pip freeze > installed_versions.txt
diff requirements.txt installed_versions.txt
# Should show no differences for pinned packages
```

### Step 3: Run Preprocessing Pipeline

```bash
# Clean any previous runs
rm -rf data/processed/*

# Run preprocessing
make preprocess
# Or: python -m mlops_online_news_popularity.cli.preprocess_cli

# Verify outputs
ls -lh data/processed/
# Should see:
# - X_train.csv, y_train.csv
# - X_val.csv, y_val.csv
# - X_test.csv, y_test.csv
# - metadata.json
```

**Expected metadata** (`data/processed/metadata.json`):
```json
{
  "train_samples": 27750,
  "val_samples": 5947,
  "test_samples": 5947,
  "n_features": 59,
  "target_col": "shares",
  "random_state": 42
}
```

### Step 4: Train Models

```bash
# Train single model (Ridge)
make train-single
# Or: python -m mlops_online_news_popularity.cli.train_cli train-single

# Or train all models
make train
# Or: python -m mlops_online_news_popularity.cli.train_cli train-compare config/models.yaml
```

### Step 5: Verify Results

**Check model artifacts**:
```bash
ls -lh models/
# Example: ridge_best_20251118_153045.pkl

# Calculate MD5 hash
md5sum models/ridge_best_*.pkl
# Note this hash for comparison
```

**Check metrics** (view in terminal output or MLflow UI):
```
Expected metrics for Ridge (approximate):
- Train RMSE: ~1.02-1.05
- Val RMSE: ~1.05-1.08
- Test RMSE: ~1.05-1.08
- R²: ~0.78-0.80
```

### Step 6: Repeat and Compare

```bash
# Clean outputs
rm -rf data/processed/* models/ridge_best_*

# Rerun pipeline
make preprocess
make train-single

# Compare hashes
md5sum models/ridge_best_*.pkl
# Should match previous hash EXACTLY

# Compare metrics in MLflow
make mlflow-ui
# Navigate to both runs, metrics should be identical
```

---

## Expected Results

### Preprocessing Pipeline

| Split | Expected Rows | Expected Features |
|-------|---------------|-------------------|
| Train | 27,750 | 59 |
| Val | 5,947 | 59 |
| Test | 5,947 | 59 |
| **Total** | **39,644** | **59** |

### Model Training (Ridge Regression)

| Metric | Expected Range |
|--------|----------------|
| **Baseline RMSE** | 1.75 - 1.85 |
| **Train RMSE** | 1.02 - 1.05 |
| **Val RMSE** | 1.05 - 1.08 |
| **Test RMSE** | 1.05 - 1.08 |
| **Train MAE** | 0.72 - 0.75 |
| **Val MAE** | 0.75 - 0.78 |
| **Test MAE** | 0.75 - 0.78 |
| **Train R²** | 0.79 - 0.82 |
| **Val R²** | 0.77 - 0.80 |
| **Test R²** | 0.77 - 0.80 |

**Note**: Exact values will be identical across runs due to random seeds.

### Model Training (RandomForest)

| Metric | Expected Range |
|--------|----------------|
| **Train RMSE** | 0.45 - 0.55 |
| **Val RMSE** | 0.95 - 1.05 |
| **Test RMSE** | 0.95 - 1.05 |
| **Train R²** | 0.92 - 0.95 |
| **Val R²** | 0.78 - 0.82 |
| **Test R²** | 0.78 - 0.82 |

---

## Artifacts Versioning

### DVC Workflow

```bash
# Track new data version
dvc add data/raw/online_news_modified.csv
git add data/raw/online_news_modified.csv.dvc
git commit -m "data: Add dataset v1.0"

# Track model version
dvc add models/
git add models.dvc
git commit -m "model: Add Ridge v1.0"

# Push to remote storage
dvc push

# Later, pull specific version
git checkout v1.0.0
dvc pull
```

### MLflow Model Registry

```bash
# Register model
mlflow.register_model(
    model_uri="runs:/<run_id>/model_pipeline",
    name="RidgeNewsPopularity"
)

# Load specific version
mlflow.pyfunc.load_model("models:/RidgeNewsPopularity/1")
```

---

## Troubleshooting

### Different Metrics on Different Machines

**Symptom**: Metrics differ by >0.01 between runs on different machines

**Possible Causes**:
1. Different Python versions
2. Different package versions
3. Different numpy/scikit-learn BLAS backends

**Solution**:
```bash
# 1. Verify Python version
python --version  # Must be 3.10.x

# 2. Reinstall from exact requirements
pip uninstall -y -r requirements.txt
pip install --no-cache-dir -r requirements.txt

# 3. Check numpy backend
python -c "import numpy; numpy.show_config()"
# Should use same BLAS library

# 4. Force CPU-only (disable GPU randomness)
export CUDA_VISIBLE_DEVICES=""
```

### Different Train/Test Splits

**Symptom**: Train/val/test split sizes differ

**Cause**: Random seed not set correctly

**Solution**:
```bash
# Verify seed in data_processor.py
grep -n "random_state" mlops_online_news_popularity/preprocessing/data_processor.py

# Should show:
# Line 43: def __init__(self, ..., random_state: int = 42):
# Line XX: train_test_split(..., random_state=self.random_state)
```

### Model File Hashes Differ

**Symptom**: MD5 hashes don't match even with same metrics

**Cause**: Pickle serialization includes metadata (timestamps)

**Solution**: Compare predictions instead of file hashes
```python
# Load both models
model1 = joblib.load("run1.pkl")
model2 = joblib.load("run2.pkl")

# Compare predictions on same data
import numpy as np
X_test = pd.read_csv("data/processed/X_test.csv")
preds1 = model1.predict(X_test)
preds2 = model2.predict(X_test)

# Should be identical
np.allclose(preds1, preds2, atol=1e-10)  # True
```

### CI Passes Locally Fails

**Symptom**: Tests pass locally but fail in GitHub Actions

**Possible Causes**:
1. Missing file in `.gitignore`
2. Different environment variables
3. Missing system dependencies

**Solution**:
```bash
# Run exactly as CI does
docker run -it --rm \
  -v $(pwd):/app \
  -w /app \
  python:3.10-slim \
  bash -c "pip install -r requirements.txt && pip install -e . && make test"
```

---

## Best Practices

### For Development

1. **Always use virtual environments**:
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate
   ```

2. **Install in editable mode**:
   ```bash
   pip install -e .
   ```

3. **Run tests before committing**:
   ```bash
   make test
   make lint
   ```

4. **Validate reproducibility periodically**:
   ```bash
   bash scripts/test_reproducibility.sh
   ```

### For Production

1. **Pin all dependencies** (already done in `requirements.txt`)

2. **Use Docker for deployment** (guarantees consistent environment):
   ```bash
   docker pull artemiop/mlops-news-predictor:v1.0.0
   ```

3. **Track experiments in MLflow**:
   - Every training run logged
   - Models registered with versions
   - Hyperparameters and metrics stored

4. **Version data with DVC**:
   - Track dataset changes
   - Enable rollback to previous versions
   - Share data across team without git bloat

### For Collaboration

1. **Document environment setup**:
   - Share this reproducibility guide
   - Include `.python-version` in repo

2. **Use consistent code style**:
   ```bash
   make format  # Before committing
   make lint    # Verify compliance
   ```

3. **Review MLflow runs**:
   - Compare metrics before merging
   - Ensure no regression

---

## Related Documentation

- [README.md](../README.md) - Project overview and quick start
- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Testing guidelines
- [Contributing](contributing.md) - Development workflow
- [Docker Deployment](deployment/dockerhub.md) - Container reproducibility

---

## Validation Checklist

Use this checklist to verify reproducibility:

- [ ] Python 3.10 confirmed (`python --version`)
- [ ] Dependencies installed from `requirements.txt` (exact versions)
- [ ] Package installed in editable mode (`pip install -e .`)
- [ ] Preprocessing produces same split sizes (27750/5947/5947)
- [ ] Model training produces metrics within expected range
- [ ] MLflow tracking URI configured correctly
- [ ] DVC data pulled successfully (if using DVC remote)
- [ ] Predictions on same data are identical across runs
- [ ] Automated test passes (`bash scripts/test_reproducibility.sh`)

---

**Last Updated**: November 2024
**Python Version**: 3.10
**Seed**: 42
