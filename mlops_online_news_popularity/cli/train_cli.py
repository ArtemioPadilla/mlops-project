"""
Model training CLI with MLflow tracking.

This script loads preprocessed data, trains models with MLflow experiment tracking,
and saves the best model based on validation metrics.

Usage:
    python -m mlops_online_news_popularity.cli.train_cli --help
    python -m mlops_online_news_popularity.cli.train_cli --config data/config.yaml
    python -m mlops_online_news_popularity.cli.train_cli --model ridge
"""

import json
from pathlib import Path

from loguru import logger
import pandas as pd
import typer

import mlflow
from mlops_online_news_popularity.config import (
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)
from mlops_online_news_popularity.modeling.compare import Experimento
from mlops_online_news_popularity.modeling.train import ModelTrainer

app = typer.Typer()


class MockDataProcessor:
    """
    Mock DataProcessor to load preprocessed data from disk.

    This class mimics the DataProcessor interface but loads data that
    has already been processed and saved to CSV files.
    """

    def __init__(self, data_dir: Path):
        """
        Load preprocessed data from directory.

        Parameters
        ----------
        data_dir : Path
            Directory containing X_train.csv, y_train.csv, etc. and metadata.json
        """
        logger.info(f"Loading preprocessed data from: {data_dir}")

        # Load splits
        self.X_train = pd.read_csv(data_dir / "X_train.csv")
        self.X_val = pd.read_csv(data_dir / "X_val.csv")
        self.X_test = pd.read_csv(data_dir / "X_test.csv")
        self.y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()
        self.y_val = pd.read_csv(data_dir / "y_val.csv").squeeze()
        self.y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()

        # Load metadata
        with open(data_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        self.cols_bin = metadata["cols_bin"]
        self.cols_no_bin = metadata["cols_no_bin"]
        self.numeric_features = metadata["numeric_features"]

        logger.info(
            f"Loaded splits: train={len(self.X_train)}, "
            f"val={len(self.X_val)}, test={len(self.X_test)}"
        )
        logger.info(
            f"Features: {len(self.cols_bin)} binary, " f"{len(self.cols_no_bin)} non-binary"
        )


@app.command()
def train_single(
    data_dir: Path = typer.Option(
        PROCESSED_DATA_DIR,
        "--data-dir",
        "-d",
        help="Directory with preprocessed data",
    ),
    model_type: str = typer.Option(
        "ridge",
        "--model",
        "-m",
        help="Model type: ridge, lasso, randomforest, xgboost",
    ),
    experiment_name: str = typer.Option(
        "model-training",
        "--experiment",
        "-e",
        help="MLflow experiment name",
    ),
    apply_log: bool = typer.Option(
        True,
        "--log-transform",
        help="Apply log(1+y) transformation to target",
    ),
):
    """
    Train a single model with MLflow tracking.

    This command:
    1. Loads preprocessed data from data/processed/
    2. Initializes MLflow tracking
    3. Trains the specified model
    4. Logs metrics, params, and model to MLflow
    """
    logger.info("=" * 70)
    logger.info("MODEL TRAINING - CLI")
    logger.info("=" * 70)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Experiment: {experiment_name}")

    # Ensure data directory exists
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.error("Run 'make preprocess' first to generate processed data")
        raise typer.Exit(code=1)

    # Load preprocessed data
    data_processor = MockDataProcessor(data_dir)

    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"MLflow experiment: {experiment_name}")

    # Create estimator based on model type
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Lasso, Ridge

    model_map = {
        "ridge": Ridge(random_state=42),
        "lasso": Lasso(random_state=42),
        "randomforest": RandomForestRegressor(n_estimators=100, random_state=42),
    }

    # Add XGBoost if available
    try:
        from xgboost import XGBRegressor

        model_map["xgboost"] = XGBRegressor(n_estimators=100, random_state=42)
    except ImportError:
        logger.warning("XGBoost not available")

    if model_type.lower() not in model_map:
        logger.error(f"Unknown model type: {model_type}")
        logger.error(f"Available models: {list(model_map.keys())}")
        raise typer.Exit(code=1)

    estimator = model_map[model_type.lower()]

    # Start MLflow run
    with mlflow.start_run(run_name=f"{model_type}-{pd.Timestamp.now():%Y%m%d-%H%M%S}"):

        # Log parameters
        mlflow.set_tag("model_type", model_type)
        mlflow.log_param("model_class", estimator.__class__.__name__)
        mlflow.log_param("target_transform", "log" if apply_log else "none")
        mlflow.log_param("n_features", data_processor.X_train.shape[1])
        mlflow.log_param("n_train", len(data_processor.X_train))
        mlflow.log_param("n_val", len(data_processor.X_val))
        mlflow.log_param("n_test", len(data_processor.X_test))

        # Create ModelTrainer
        logger.info(f"\nTraining {model_type}...")
        trainer = ModelTrainer(
            data_processor=data_processor,
            estimator=estimator,
            model_name=model_type.upper(),
        )

        # Transform target
        if apply_log:
            trainer.transform_target(apply_log=True)
            mlflow.log_metric("baseline_rmse", trainer.baseline_rmse)

        # Train model
        trainer.train_model()

        # Evaluate model
        metrics = trainer.evaluate_model()

        # Log metrics to MLflow
        for split in ["train", "val", "test"]:
            for metric_name, value in metrics[split].items():
                mlflow.log_metric(f"{split}_{metric_name}", value)

        # Log gaps
        for gap_name, value in metrics["gaps"].items():
            mlflow.log_metric(gap_name, value)

        # Save model
        mlflow.sklearn.log_model(trainer.pipeline, "model_pipeline")
        logger.success("Model logged to MLflow")

        # Save model locally
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"{model_type}_{pd.Timestamp.now():%Y%m%d_%H%M%S}.pkl"

        import joblib

        joblib.dump(trainer.pipeline, model_path)
        mlflow.log_artifact(str(model_path), "local_model")
        logger.success(f"Model saved locally: {model_path}")

    logger.info("\n" + "=" * 70)
    logger.success("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Val RMSE: {metrics['val']['rmse']:.4f}")
    logger.info(f"Val R²: {metrics['val']['r2']:.4f}")
    logger.info(f"Test RMSE: {metrics['test']['rmse']:.4f}")
    logger.info(f"Test R²: {metrics['test']['r2']:.4f}")
    logger.info("=" * 70)


@app.command()
def train_compare(
    config_path: Path = typer.Argument(
        ...,
        help="Path to YAML config file with models to compare",
    ),
    data_dir: Path = typer.Option(
        PROCESSED_DATA_DIR,
        "--data-dir",
        "-d",
        help="Directory with preprocessed data",
    ),
):
    """
    Train and compare multiple models using config file.

    This command uses the Experimento class to train multiple models
    defined in a YAML configuration file and compare them in MLflow.

    Example config.yaml:
        experiment_name: "Model Comparison"
        metric_to_optimize: "val_rmse"
        optimize_mode: "ASC"
        models_to_try:
          Ridge:
            class_path: "sklearn.linear_model.Ridge"
          RandomForest:
            class_path: "sklearn.ensemble.RandomForestRegressor"
    """
    logger.info("=" * 70)
    logger.info("MODEL COMPARISON - CLI")
    logger.info("=" * 70)
    logger.info(f"Config: {config_path}")
    logger.info(f"Data directory: {data_dir}")

    # Ensure files exist
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        raise typer.Exit(code=1)

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.error("Run 'make preprocess' first to generate processed data")
        raise typer.Exit(code=1)

    # Load preprocessed data
    data_processor = MockDataProcessor(data_dir)

    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Create experiment and run comparison
    experiment = Experimento(
        config_path=str(config_path),
        data_processor=data_processor,
    )

    # Run all experiments
    experiment.ejecuta_experimentos()

    # Get best model
    best_model = experiment.mejor_modelo()

    logger.info("\n" + "=" * 70)
    logger.success("MODEL COMPARISON COMPLETE")
    logger.info("=" * 70)
    if best_model:
        logger.info(f"Best model: {best_model['model_name']}")
        logger.info(f"Best {best_model['metric_name']}: {best_model['score']:.4f}")
        logger.info(f"MLflow URI: {best_model['model_uri']}")
    logger.info("=" * 70)


if __name__ == "__main__":
    app()
