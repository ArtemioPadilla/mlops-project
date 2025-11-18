"""
ModelHandler for serving trained ML models.

Implements the handler pattern with initialize, preprocess, inference,
and postprocess methods similar to AWS SageMaker handlers.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

import joblib
from loguru import logger
import mlflow
import numpy as np
import pandas as pd

from mlops_online_news_popularity.serving import config


class ModelHandler:
    """
    Handler for model inference with preprocessing and postprocessing.

    This class implements a pattern similar to AWS SageMaker handlers,
    with separate methods for each stage of the inference pipeline.
    """

    def __init__(self):
        """Initialize the ModelHandler."""
        start = time.time()
        self.initialized = False
        self.model = None
        self.model_name = None
        self.model_info = {}
        logger.info(f"ModelHandler __init__ took {(time.time() - start) * 1000:.2f} ms")

    def initialize(self, context: Optional[Dict[str, Any]] = None):
        """
        Initialize the model by loading from MLflow or local file.

        Args:
            context: Optional context dictionary with model loading parameters.
                    Can override MODEL_LOAD_STRATEGY, MODEL_PATH, MLFLOW_RUN_ID, etc.
        """
        start = time.time()

        # Override config with context if provided
        load_strategy = (
            context.get("load_strategy", config.MODEL_LOAD_STRATEGY)
            if context
            else config.MODEL_LOAD_STRATEGY
        )
        model_path = context.get("model_path", config.MODEL_PATH) if context else config.MODEL_PATH
        mlflow_run_id = (
            context.get("mlflow_run_id", config.MLFLOW_RUN_ID) if context else config.MLFLOW_RUN_ID
        )

        try:
            if load_strategy == "mlflow":
                self._load_from_mlflow(mlflow_run_id)
            else:
                self._load_from_local(model_path)

            self.initialized = True
            logger.info(
                f"ModelHandler initialized successfully in {(time.time() - start) * 1000:.2f} ms"
            )
            logger.info(f"Model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize ModelHandler: {e}")
            raise

    def _load_from_local(self, model_path: str):
        """Load model from local pickle file."""
        logger.info(f"Loading model from local path: {model_path}")
        start = time.time()

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load the model
        self.model = joblib.load(model_path)
        self.model_name = path.stem
        self.model_info = {
            "model_name": self.model_name,
            "model_path": str(path),
            "load_strategy": "local",
            "model_size_mb": path.stat().st_size / (1024 * 1024),
        }

        logger.info(f"Model loaded from local file in {(time.time() - start) * 1000:.2f} ms")

    def _load_from_mlflow(self, run_id: Optional[str] = None):
        """Load model from MLflow."""
        logger.info(f"Loading model from MLflow (run_id: {run_id})")
        start = time.time()

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

        if not run_id:
            raise ValueError("MLFLOW_RUN_ID must be provided when using mlflow load strategy")

        # Load model from MLflow
        model_uri = f"runs:/{run_id}/model_pipeline"
        self.model = mlflow.sklearn.load_model(model_uri)

        # Get run info
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)

        self.model_name = run.data.params.get("model_name", "unknown")
        self.model_info = {
            "model_name": self.model_name,
            "run_id": run_id,
            "load_strategy": "mlflow",
            "experiment_id": run.info.experiment_id,
            "metrics": run.data.metrics,
            "params": run.data.params,
        }

        logger.info(f"Model loaded from MLflow in {(time.time() - start) * 1000:.2f} ms")

    def preprocess(self, input_data: Union[pd.DataFrame, Dict, List[Dict]]) -> pd.DataFrame:
        """
        Preprocess input data for inference.

        The sklearn Pipeline in the model already handles scaling and transformations,
        so this method mainly validates and formats the input.

        Args:
            input_data: Input data as DataFrame, dict, or list of dicts

        Returns:
            Preprocessed DataFrame ready for inference
        """
        start = time.time()

        # Convert input to DataFrame if needed
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise ValueError(
                f"Unsupported input type: {type(input_data)}. "
                "Expected DataFrame, dict, or list of dicts"
            )

        # Validate features
        missing_features = set(config.FEATURE_NAMES) - set(df.columns)
        if missing_features:
            raise ValueError(
                f"Missing required features: {missing_features}. "
                f"Expected {len(config.FEATURE_NAMES)} features: {config.FEATURE_NAMES}"
            )

        # Select only the required features in the correct order
        df = df[config.FEATURE_NAMES]

        # Check for any NaN values that might cause issues
        if df.isnull().any().any():
            logger.warning("Input contains NaN values. Model pipeline will handle imputation.")

        logger.debug(f"Preprocessing took {(time.time() - start) * 1000:.2f} ms")
        return df

    def inference(self, inputs: pd.DataFrame) -> np.ndarray:
        """
        Run inference on preprocessed inputs.

        Args:
            inputs: Preprocessed DataFrame

        Returns:
            Model predictions (log-transformed shares)
        """
        start = time.time()

        if not self.initialized:
            raise RuntimeError("ModelHandler not initialized. Call initialize() first.")

        # Make predictions
        predictions = self.model.predict(inputs)

        logger.debug(
            f"Inference on {len(inputs)} samples took {(time.time() - start) * 1000:.2f} ms"
        )
        return predictions

    def postprocess(self, predictions: np.ndarray) -> List[Dict[str, float]]:
        """
        Postprocess predictions.

        Applies inverse log transformation (expm1) to convert log-transformed
        predictions back to actual share counts.

        Args:
            predictions: Raw model predictions (log-transformed)

        Returns:
            List of dictionaries with predictions and metadata
        """
        start = time.time()

        # Apply inverse log transformation: shares = exp(log1p(shares)) - 1
        # Which simplifies to: shares = expm1(predictions)
        actual_shares = np.expm1(predictions)

        # Round to nearest integer (shares are counts)
        actual_shares = np.round(actual_shares).astype(int)

        # Ensure non-negative (shares can't be negative)
        actual_shares = np.maximum(actual_shares, 0)

        # Format output
        results = [
            {
                "predicted_shares": int(share),
                "log_prediction": float(log_pred),
            }
            for share, log_pred in zip(actual_shares, predictions)
        ]

        logger.debug(f"Postprocessing took {(time.time() - start) * 1000:.2f} ms")
        return results

    def handle(self, input_data: Union[pd.DataFrame, Dict, List[Dict]]) -> List[Dict[str, float]]:
        """
        Handle complete inference pipeline: preprocess -> inference -> postprocess.

        Args:
            input_data: Input data as DataFrame, dict, or list of dicts

        Returns:
            List of prediction results
        """
        start = time.time()

        # Preprocess
        preprocessed = self.preprocess(input_data)

        # Inference
        predictions = self.inference(preprocessed)

        # Postprocess
        results = self.postprocess(predictions)

        logger.info(
            f"Complete handle pipeline for {len(results)} samples "
            f"took {(time.time() - start) * 1000:.2f} ms"
        )

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata.

        Returns:
            Dictionary with model information
        """
        if not self.initialized:
            return {"status": "not_initialized", "message": "Model not loaded"}

        return {
            "status": "ready",
            "model_info": self.model_info,
            "features": {
                "count": len(config.FEATURE_NAMES),
                "names": config.FEATURE_NAMES,
            },
            "target": config.TARGET_COL,
        }


# Singleton instance
_service = ModelHandler()


def get_model_handler() -> ModelHandler:
    """
    Get the singleton ModelHandler instance.

    Returns:
        ModelHandler instance
    """
    return _service
