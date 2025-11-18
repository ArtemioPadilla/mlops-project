"""
Configuration for the serving module.

Handles environment variables and settings for model loading and API configuration.
"""

import os
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJ_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJ_ROOT / "models"
MLFLOW_ARTIFACTS_DIR = PROJ_ROOT / "mlflow_artifacts"

# Model configuration
MODEL_NAME = os.getenv("MODEL_NAME", "RandomForestBase")
MODEL_LOAD_STRATEGY: Literal["local", "mlflow"] = os.getenv(
    "MODEL_LOAD_STRATEGY", "local"
)

# Local model path (default to latest RandomForest)
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(MODELS_DIR / "randomforestbase_best_20251102_165526.pkl")
)

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{MLFLOW_ARTIFACTS_DIR}/dev/mlflow.db"
)
MLFLOW_RUN_ID: Optional[str] = os.getenv("MLFLOW_RUN_ID", None)
MLFLOW_MODEL_NAME: Optional[str] = os.getenv("MLFLOW_MODEL_NAME", None)

# Server configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_RELOAD = os.getenv("API_RELOAD", "false").lower() == "true"

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Feature list (59 features from metadata.json)
FEATURE_NAMES = [
    "n_tokens_title",
    "n_tokens_content",
    "n_unique_tokens",
    "n_non_stop_words",
    "n_non_stop_unique_tokens",
    "num_hrefs",
    "num_self_hrefs",
    "num_imgs",
    "num_videos",
    "average_token_length",
    "num_keywords",
    "data_channel_is_lifestyle",
    "data_channel_is_entertainment",
    "data_channel_is_bus",
    "data_channel_is_socmed",
    "data_channel_is_tech",
    "data_channel_is_world",
    "kw_min_min",
    "kw_max_min",
    "kw_avg_min",
    "kw_min_max",
    "kw_max_max",
    "kw_avg_max",
    "kw_min_avg",
    "kw_max_avg",
    "kw_avg_avg",
    "self_reference_min_shares",
    "self_reference_max_shares",
    "self_reference_avg_sharess",
    "weekday_is_monday",
    "weekday_is_tuesday",
    "weekday_is_wednesday",
    "weekday_is_thursday",
    "weekday_is_friday",
    "weekday_is_saturday",
    "weekday_is_sunday",
    "is_weekend",
    "LDA_00",
    "LDA_01",
    "LDA_02",
    "LDA_03",
    "LDA_04",
    "global_subjectivity",
    "global_sentiment_polarity",
    "global_rate_positive_words",
    "global_rate_negative_words",
    "rate_positive_words",
    "rate_negative_words",
    "avg_positive_polarity",
    "min_positive_polarity",
    "max_positive_polarity",
    "avg_negative_polarity",
    "min_negative_polarity",
    "max_negative_polarity",
    "title_subjectivity",
    "title_sentiment_polarity",
    "abs_title_subjectivity",
    "abs_title_sentiment_polarity",
    "mixed_type_col",
]

# Target variable
TARGET_COL = "shares"
