# Environment Variables

Configuration via environment variables.

## .env File

Create `.env` in project root:

```bash
# MLflow tracking
MLFLOW_TRACKING_URI=sqlite:///mlflow/dev/mlflow.db

# Optional: Other settings
LOG_LEVEL=DEBUG
```

## Supported Variables

### MLFLOW_TRACKING_URI

Override default MLflow tracking URI.

**Default**: `sqlite:///mlflow/dev/mlflow.db`

**Examples**:
```bash
# Use quickstart environment
MLFLOW_TRACKING_URI=sqlite:///mlflow/quickstart/mlflow.db

# Use remote server
MLFLOW_TRACKING_URI=http://mlflow-server:5000
```

## Loading

The `.env` file is automatically loaded by `config.py`:

```python
from dotenv import load_dotenv
load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", MLFLOW_DEV_URI)
```
