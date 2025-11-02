# Troubleshooting Guide

Common issues and their solutions.

## Installation Issues

### ModuleNotFoundError: No module named 'mlops_online_news_popularity'

**Symptom**:
```python
ImportError: No module named 'mlops_online_news_popularity'
```

**Cause**: Package not installed in editable mode.

**Solution**:
```bash
# From project root
pip install -e .
```

!!! tip
    Always run `pip install -e .` after cloning the repository. This makes the `mlops_online_news_popularity` package importable.

---

### Import Error: 'ydata_profiling' not found

**Symptom**:
```
Warning: ydata-profiling no está instalado
```

**Solution**:
```bash
pip install ydata-profiling
```

**Alternative**: The pipeline will skip profiling reports if ydata-profiling isn't installed. This doesn't affect the core functionality.

---

## Data Issues

### FileNotFoundError: data/raw/online_news_modified.csv

**Cause**: Raw data file missing.

**Solution 1** - If using DVC:
```bash
dvc pull
```

**Solution 2** - Manual download:
1. Obtain the dataset from the source
2. Place it in `data/raw/online_news_modified.csv`

---

### ValueError: columns have different shapes

**Cause**: Mismatch between training and inference data schemas.

**Solution**:
- Ensure the same preprocessing is applied
- Use the saved pipeline for predictions:
```python
import joblib
pipeline = joblib.load("models/best_model.pkl")
predictions = pipeline.predict(X_new)  # Pipeline handles preprocessing
```

---

## ML flow Issues

### MLflow UI shows no experiments

**Cause**: No experiments have been run yet.

**Solution**:
```bash
# Train models first
make train

# Then start UI
make mlflow-ui
```

---

### MLflowException: Run ID not found

**Cause**: The MLflow database was deleted or the run doesn't exist.

**Solution**:
```bash
# Reset MLflow tracking (WARNING: deletes all experiments)
rm -rf mlflow_artifacts/dev/

# Re-run training
make train
```

---

### Wrong MLflow tracking URI

**Symptom**: Experiments not appearing in expected database.

**Solution**: Check your tracking URI:
```python
from mlops_online_news_popularity.config import MLFLOW_TRACKING_URI
print(MLFLOW_TRACKING_URI)
# Should show: sqlite:///mlflow_artifacts/dev/mlflow.db
```

Override via `.env` file if needed:
```bash
# .env
MLFLOW_TRACKING_URI=sqlite:///mlflow_artifacts/dev/mlflow.db
```

---

## Model Training Issues

### ConvergenceWarning: Increase max_iter

**Symptom**:
```
ConvergenceWarning: Solver terminated with max_iter=1000 without converging
```

**Solution**: Not critical, but you can increase iterations:
```yaml
# config/models.yaml
Ridge:
  class_path: "sklearn.linear_model.Ridge"
  max_iter: 5000  # Add model parameters
```

---

### Memory Error during training

**Cause**: Dataset too large for available RAM.

**Solutions**:
1. **Reduce dataset size** (for testing):
```python
# In preprocess_cli.py, add sampling
df = df.sample(frac=0.1, random_state=42)
```

2. **Use incremental learning**:
```python
# Use SGDRegressor instead of Ridge
from sklearn.linear_model import SGDRegressor
```

3. **Increase swap space** (Linux/Mac)

---

## DVC Issues

### dvc pull fails

**Cause**: No remote storage configured.

**Solution**: Configure DVC remote:
```bash
# Add remote storage (example: local directory)
dvc remote add -d myremote /path/to/dvc/storage

# Or use S3
dvc remote add -d myremote s3://mybucket/dvcstore
```

---

### Permission denied on .dvc/config

**Cause**: `.dvc/config` is gitignored for security (may contain credentials).

**Solution**: Use `.dvc/config.example` as template:
```bash
cp .dvc/config.example .dvc/config
# Edit .dvc/config with your settings
```

---

## Performance Issues

### Preprocessing takes too long

**Solutions**:

1. **Disable profiling reports** (saves time):
```python
# In data_processor.py, comment out:
# DataExplorer.generate_profiling_report(...)
```

2. **Use multiprocessing** for correlation calculation:
```python
# Enable parallel correlation calculation
corr_matrix = df.corr(method='pearson', numeric_only=True)
```

---

### Training very slow

**Causes & Solutions**:

1. **RandomForest with too many trees**:
```yaml
# config/models.yaml
RandomForest:
  class_path: "sklearn.ensemble.RandomForestRegressor"
  n_estimators: 50  # Reduce from default 100
  n_jobs: -1  # Use all CPU cores
```

2. **GridSearch over many parameters**: Use fewer parameter combinations.

3. **Large dataset**: Sample the data for experimentation:
```python
X_train_sample = X_train.sample(frac=0.1, random_state=42)
```

---

## Code Quality Issues

### Black/flake8/isort failures

**Solution**: Auto-fix most issues:
```bash
make format  # Runs black and isort
make lint    # Check remaining issues
```

**Common flake8 errors**:
- `E501`: Line too long (max 99) → Run `black` to auto-fix
- `F401`: Unused import → Remove the import
- `E731`: Lambda assignment → Convert to def function

---

## Git/GitHub Issues

### Large files rejected by GitHub

**Cause**: Trying to commit data files directly.

**Solution**: Use DVC for data:
```bash
# Never commit data directly
git rm --cached data/raw/large_file.csv

# Track with DVC instead
dvc add data/raw/large_file.csv
git add data/raw/large_file.csv.dvc
git commit -m "Track large file with DVC"
```

---

## Documentation Issues

### MkDocs build fails

**Symptom**:
```
Error: Config file 'mkdocs.yml' does not exist
```

**Solution**: Ensure you're in the project root:
```bash
cd /path/to/mlops-project
mkdocs build
```

---

### Mermaid diagrams not rendering

**Cause**: mermaid2 plugin not installed.

**Solution**:
```bash
pip install mkdocs-mermaid2-plugin
mkdocs build
```

---

## Testing Issues

### pytest: No tests found

**Cause**: Tests not discovered.

**Solution**: Ensure test files follow naming convention:
- Files: `test_*.py` or `*_test.py`
- Functions: `test_*()`
- Classes: `Test*`

```bash
# Run from project root
pytest tests/

# Verbose mode
pytest -v tests/
```

---

## Getting Help

If your issue isn't listed here:

1. **Check logs**: Look for detailed error messages
2. **Enable debug logging**:
```python
from loguru import logger
logger.add("debug.log", level="DEBUG")
```

3. **Check GitHub Issues**: [github.com/artemiopadilla/mlops-project/issues](https://github.com/artemiopadilla/mlops-project/issues)

4. **Review documentation**: Ensure you followed all setup steps correctly
