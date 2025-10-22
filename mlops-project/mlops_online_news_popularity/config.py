from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# MLflow tracking
MLFLOW_DIR = PROJ_ROOT / "mlflow"
MLFLOW_QUICKSTART_DIR = MLFLOW_DIR / "quickstart"
MLFLOW_DEV_DIR = MLFLOW_DIR / "dev"

# Ensure MLflow directories exist
MLFLOW_QUICKSTART_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_DEV_DIR.mkdir(parents=True, exist_ok=True)

# MLflow tracking URIs
MLFLOW_QUICKSTART_URI = f"sqlite:///{MLFLOW_QUICKSTART_DIR / 'mlflow.db'}"
MLFLOW_DEV_URI = f"sqlite:///{MLFLOW_DEV_DIR / 'mlflow.db'}"

# Allow environment variable override for MLflow tracking URI
import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", MLFLOW_DEV_URI)

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
