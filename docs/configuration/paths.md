# Path Management

Centralized configuration in `config.py`.

## Path Constants

```python
from mlops_online_news_popularity.config import (
    PROJ_ROOT,          # Project root directory
    DATA_DIR,           # data/
    RAW_DATA_DIR,       # data/raw/
    PROCESSED_DATA_DIR, # data/processed/
    MODELS_DIR,         # models/
    REPORTS_DIR,        # reports/
    FIGURES_DIR,        # reports/figures/
    DOCS_DIR,           # docs/
    PROFILING_REPORTS_DIR,  # docs/assets/html/
    PROFILING_IMAGES_DIR,   # docs/assets/images/
    MLFLOW_DEV_DIR,     # mlflow_artifacts/dev/
    MLFLOW_QUICKSTART_DIR,  # mlflow_artifacts/quickstart/
)
```

## Usage

```python
# Good: Use path constants
from mlops_online_news_popularity.config import RAW_DATA_DIR
df = pd.read_csv(RAW_DATA_DIR / "dataset.csv")

# Bad: Hardcode paths
df = pd.read_csv("data/raw/dataset.csv")  # Don't do this
```

## MLflow URIs

```python
from mlops_online_news_popularity.config import (
    MLFLOW_TRACKING_URI,    # Current tracking URI
    MLFLOW_DEV_URI,         # Dev environment
    MLFLOW_QUICKSTART_URI,  # Quickstart environment
)
```

## Environment Override

Create `.env` file to override:

```bash
# .env
MLFLOW_TRACKING_URI=sqlite:///mlflow_artifacts/quickstart/mlflow.db
```
