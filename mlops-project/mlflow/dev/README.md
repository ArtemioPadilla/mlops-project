# MLflow Development Environment

This directory contains the MLflow tracking database and artifacts for **actual model development** work.

## Purpose

Use this environment for:
- Real model training experiments
- Model versioning and comparison
- Pre-production model evaluation
- Tracking production-candidate models

## Contents

- `mlflow.db` - SQLite tracking database (gitignored)
- `mlruns/` - MLflow artifact storage directory (gitignored)
- `artifacts/` - Additional artifact storage (gitignored)

## Usage

```python
from mlops_online_news_popularity.config import MLFLOW_DEV_URI
import mlflow

mlflow.set_tracking_uri(MLFLOW_DEV_URI)
mlflow.set_experiment("my-experiment-name")

# Your training code here...
```

Or use the default tracking URI from config:

```python
from mlops_online_news_popularity.config import MLFLOW_TRACKING_URI
import mlflow

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)  # Uses dev by default
```

## View MLflow UI

From the project root:

```bash
cd mlops-project
mlflow ui --backend-store-uri sqlite:///mlflow/dev/mlflow.db
```

Then open http://localhost:5000 in your browser.

## Notes

- This environment persists across sessions - don't delete unless you want to lose your experiment history
- The database and artifacts are gitignored but should be backed up if they contain important experiments
- Consider switching to a remote tracking server when moving to production
