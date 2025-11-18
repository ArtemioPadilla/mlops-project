"""
Model comparison module for MLflow experiment orchestration.

This module provides the Experimento class for comparing multiple regression models
using MLflow for experiment tracking and model comparison.
"""

from datetime import datetime
import importlib
import inspect
from pathlib import Path
from typing import Any, Dict
from mlops_online_news_popularity.seeds import set_global_seed, SEED

import joblib
from loguru import logger
import mlflow
from mlflow import MlflowClient
import yaml

from mlops_online_news_popularity.config import MODELS_DIR
from mlops_online_news_popularity.modeling.train import ModelTrainer
from mlops_online_news_popularity.preprocessing import DataProcessor

# =============================================================================
# Requirements and Usage Instructions
# =============================================================================
# 1. Import the required classes:
#      from mlops_online_news_popularity.preprocessing import DataProcessor
#      from mlops_online_news_popularity.modeling.train import ModelTrainer
#      from mlops_online_news_popularity.modeling.compare import Experimento
#
# 2. Prepare data using DataProcessor (model-agnostic preprocessing):
#       processor = DataProcessor(
#           filepath='data/raw/online_news_modified.csv',
#           target_col='shares'
#       )
#       processor.process()
#
# 3. Create the Experimento object with the processed data:
#       experiment = Experimento(
#           config_path="config/models.yaml",
#           data_processor=processor
#       )
#
# 4. Train and evaluate all models defined in YAML:
#       experiment.ejecuta_experimentos()
#
# 5. Get the best model from MLflow:
#       best_model_info = experiment.mejor_modelo()


class Experimento:
    """
    Orchestrate multi-model comparison experiments using MLflow.

    This class uses DataProcessor (for model-agnostic preprocessing) and ModelTrainer
    (for model-specific training) to train multiple regression models and compare them.
    Results are stored in MLflow for easy comparison and selection of the best model.

    YAML Configuration Format:
    --------------------------
    experiment_name: "Model Comparison"
    metric_to_optimize: "val_rmse"  # or "val_r2", "test_rmse", etc.
    optimize_mode: "ASC"  # ASC for RMSE/MAE, DESC for R²

    models_to_try:
      Ridge:
        class_path: "sklearn.linear_model.Ridge"
      RandomForest:
        class_path: "sklearn.ensemble.RandomForestRegressor"

    Note: Scaling is handled automatically by ModelTrainer based on column types.
    """

    def __init__(self, config_path: str, data_processor: DataProcessor, seed: int = SEED):
        """
        Initialize the experiment with configuration and data.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file
        data_processor : DataProcessor
            DataProcessor object containing train/val/test splits
        seed : int
            Global seed
        """
        self.config_path = config_path
        self.data = data_processor
        self.seed = set_global_seed(seed)

        # Import the YAML configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.experiment_name = self.config["experiment_name"]
        self.models_config = self.config["models_to_try"]

        # Initialize MLflow with deleted experiment handling
        self._setup_mlflow_experiment()
        logger.info(f"MLflow: Experiment to run '{self.experiment_name}'")
        print(f"MLflow: Experimento a Correr '{self.experiment_name}'")

    def _setup_mlflow_experiment(self) -> None:
        """
        Set up MLflow experiment with proper handling of deleted experiments.

        If the experiment exists but is deleted, it will be automatically restored.
        This prevents the common error: "Cannot set a deleted experiment as active".
        """
        client = MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)

        if experiment is not None and experiment.lifecycle_stage == "deleted":
            logger.warning(
                f"Experiment '{self.experiment_name}' (ID: {experiment.experiment_id}) "
                f"is deleted. Restoring it automatically..."
            )
            print(
                f"⚠️  El experimento '{self.experiment_name}' fue eliminado previamente. "
                f"Restaurándolo automáticamente..."
            )

            # Restore the deleted experiment
            client.restore_experiment(experiment.experiment_id)

            logger.success(f"Experiment '{self.experiment_name}' restored successfully")
            print(f"✓ Experimento '{self.experiment_name}' restaurado exitosamente")

        # Now safely set the experiment (either existing active or newly restored)
        mlflow.set_experiment(self.experiment_name)

    def _instantiate_model(self, class_path: str) -> Any:
        """
        Import and generate an instance of the model.

        Parameters
        ----------
        class_path : str
            Full module path to the model class (e.g., 'sklearn.linear_model.Ridge')

        Returns
        -------
        Any
            Instantiated model object
        """
        try:
            module_name, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            EstimatorClass = getattr(module, class_name)

            # MLOps: insert "random_state" to reproduce the model if the model supports it
            init_kwargs = {}
            if "random_state" in inspect.signature(EstimatorClass.__init__).parameters:
                init_kwargs["random_state"] = self.seed

            return EstimatorClass(**init_kwargs)
        except Exception as e:
            logger.error(f"Error de instancia {class_path}: {e}")
            print(f"Error de instancia {class_path}: {e}")
            raise

    def ejecuta_experimentos(self) -> None:
        """
        Run all models specified in the config file and compare them.

        This creates a parent run in MLflow and nested child runs for each model.
        Each model is trained, evaluated, and logged to MLflow.
        """
        set_global_seed(self.seed)

        # Create unique run name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        num_models = len(self.models_config)
        parent_run_name = f"{self.experiment_name}-{num_models}models-{timestamp}"

        # Create the parent process for MLFLOW which contains the entire code execution
        with mlflow.start_run(run_name=parent_run_name) as parent_run:
            # Log configuration metadata
            mlflow.log_param("config_file", self.config_path)
            mlflow.log_param("experiment_name", self.experiment_name)
            mlflow.log_param("seed_used", self.seed)
            mlflow.log_param("num_models", num_models)
            mlflow.log_param("models_trained", ",".join(self.models_config.keys()))
            mlflow.log_param("metric_to_optimize", self.config.get("metric_to_optimize"))
            mlflow.log_param("optimize_mode", self.config.get("optimize_mode"))

            # Log global preprocessing summary
            mlflow.log_param("data_source", self.data.filepath)
            mlflow.log_param("total_features", self.data.X_train.shape[1])
            mlflow.log_param(
                "total_samples",
                len(self.data.X_train) + len(self.data.X_val) + len(self.data.X_test),
            )
            mlflow.log_param(
                "preprocessing_correlation_threshold", self.data.correlation_threshold
            )
            logger.info("Logged global preprocessing configuration to parent run")

            # Add tags for filtering
            mlflow.set_tag("run_type", "parent")
            mlflow.set_tag("num_models", num_models)
            mlflow.set_tag("timestamp", timestamp)

            logger.info(f"Parent process started for MLFLOW: {parent_run.info.run_id}")
            print(f"Proceso Padre Iniciado para MLFLOW: {parent_run.info.run_id}")
            print("Comenzando Evaluacion y Comparación")

            for model_name, model_cfg in self.models_config.items():

                logger.info(f"Running model: {model_name}")
                print(f"Ejecutando modelo : {model_name} ")

                # Creating first thread/child
                with mlflow.start_run(run_name=model_name, nested=True) as child_run:

                    try:
                        # Creating instance
                        estimator = self._instantiate_model(model_cfg["class_path"])

                        # Add MLFlow tags
                        mlflow.set_tag("run_type", "child")
                        mlflow.set_tag("model_name", model_name)
                        mlflow.log_param("class_path", model_cfg["class_path"])
                        mlflow.log_param("target_transform", "log")  # Always log transform

                        # Log all model hyperparameters
                        model_params = estimator.get_params(deep=True)
                        loggable_params = {
                            f"model_{k}": v
                            for k, v in model_params.items()
                            if isinstance(v, (int, float, str, bool, type(None)))
                        }
                        mlflow.log_params(loggable_params)
                        logger.info(f"Logged {len(loggable_params)} model hyperparameters")

                        # Log DataProcessor configuration
                        mlflow.log_param("data_filepath", self.data.filepath)
                        mlflow.log_param("target_col", self.data.target_col)
                        mlflow.log_param("correlation_threshold", self.data.correlation_threshold)
                        mlflow.log_param("cols_to_drop", ",".join(self.data.cols_to_drop))
                        if self.data.cols_dropped_correlation:
                            mlflow.log_param(
                                "cols_dropped_correlation",
                                ",".join(self.data.cols_dropped_correlation),
                            )
                        else:
                            mlflow.log_param("cols_dropped_correlation", "none")

                        # Log dataset sizes and feature information
                        mlflow.log_param("train_size", len(self.data.X_train))
                        mlflow.log_param("val_size", len(self.data.X_val))
                        mlflow.log_param("test_size", len(self.data.X_test))
                        mlflow.log_param("num_features", self.data.X_train.shape[1])
                        mlflow.log_param("num_binary_features", len(self.data.cols_bin))
                        mlflow.log_param("num_nonbinary_features", len(self.data.cols_no_bin))
                        logger.info("Logged DataProcessor configuration and dataset statistics")

                        # Creating ModelTrainer object to train and evaluate models
                        trainer = ModelTrainer(
                            data_processor=self.data,
                            estimator=estimator,
                            model_name=model_name
                        )

                        # Transform target if needed (log transformation for skewed data)
                        trainer.transform_target(apply_log=True)

                        # Log preprocessing pipeline details
                        mlflow.log_param("imputation_strategy_nonbinary", "median")
                        mlflow.log_param("imputation_strategy_binary", "most_frequent")
                        mlflow.log_param("power_transform_method", "yeo-johnson")
                        mlflow.log_param("scaling_method", "standard")
                        if trainer.baseline_rmse is not None:
                            mlflow.log_metric("baseline_rmse", trainer.baseline_rmse)
                        logger.info("Logged preprocessing pipeline configuration")

                        # Training and evaluating the model
                        trainer.train_model()
                        # Evaluate the trained model
                        metrics = trainer.evaluate_model()

                        # Save results in MLFlow
                        mlflow.log_metric("train_rmse", metrics["train"]["rmse"])
                        mlflow.log_metric("train_r2", metrics["train"]["r2"])
                        mlflow.log_metric("val_rmse", metrics["val"]["rmse"])
                        mlflow.log_metric("val_r2", metrics["val"]["r2"])
                        mlflow.log_metric("test_rmse", metrics["test"]["rmse"])
                        mlflow.log_metric("test_r2", metrics["test"]["r2"])

                        # Save the model
                        mlflow.sklearn.log_model(trainer.pipeline, "model_pipeline")

                        logger.success(
                            f"MLflow: {model_name} registered (Run ID: {child_run.info.run_id})"
                        )
                        print(f"MLflow: {model_name} registrado (Run ID: {child_run.info.run_id})")

                    except Exception as e:
                        logger.error(f"Error training model {model_name}: {e}")
                        print(f"Error al entrenar el modelo {model_name}: {e}")
                        mlflow.set_tag("status", "FAILED")
                        mlflow.log_param("error", str(e))

            logger.success("Model comparison completed")
            print("\nComparacion de Modelos Terminada")
            mlflow.set_tag("status", "SUCCESS")

    # -------------------------------------------------------------------------
    # Function that obtains the best model by comparing the contents in
    # the config file previously saved in MLFLOW
    # -------------------------------------------------------------------------
    def mejor_modelo(self) -> Dict[str, Any]:
        """
        Connect with MLFlow to review which model is considered the best in the
        experiment run from the config file.

        Also saves the best model locally to the models/ directory.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing best model information:
            - model_name: Name of the best model
            - metric_name: Metric used for optimization
            - score: Score of the best model
            - run_id: MLflow run ID
            - model_uri: URI to load the model
        """
        logger.info("Comparing results to obtain the best model in MLFlow")
        print("\nComparando Resultados para obtencion del mejor modelo en MLFlow")

        # Get metric to optimize (use underscore format for MLflow order_by)
        metric_to_optimize = self.config["metric_to_optimize"]
        order_mode = self.config["optimize_mode"]

        # Validate and normalize optimize_mode for MLflow
        order_mode_map = {"ASCENDING": "ASC", "ASC": "ASC", "DESCENDING": "DESC", "DESC": "DESC"}
        order_mode = order_mode_map.get(order_mode.upper(), "ASC")
        logger.info(f"Using order mode: {order_mode}")

        # Get the experiment ID to evaluate
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not experiment:
            logger.error("No experiment ID found, check saved data or ID")
            print("No se encontro el ID, revisar datos guardados o el ID")
            return {}

        # Find the latest parent run (most recent execution)
        # This ensures we only compare models from the current run, not old ones
        parent_runs_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.run_type = 'parent'",
            order_by=["start_time DESC"],
            max_results=1,
        )

        if parent_runs_df.empty:
            logger.error("No parent run found in experiment")
            print("No se encontró ningún parent run en el experimento")
            return {}

        latest_parent_id = parent_runs_df.iloc[0]["run_id"]
        logger.info(f"Searching models from latest parent run: {latest_parent_id}")

        # Search for child runs ONLY from the latest parent execution (use dotted format for MLflow)
        best_run_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{latest_parent_id}'",
            order_by=[f"metrics.{metric_to_optimize} {order_mode}"],
            max_results=1,  # search for the best model
        )

        if best_run_df.empty:
            logger.warning("No child runs found for the latest parent")
            print("No se encontraron child runs para el parent más reciente")
            return {}

        # Export the information of the best result
        best_run_data = best_run_df.iloc[0]
        best_model_name = best_run_data["tags.model_name"]
        best_run_id = best_run_data["run_id"]
        # Use underscore format for DataFrame access
        best_metric_score = best_run_data[f"metrics.{metric_to_optimize}"]

        # Save the address of the best model found
        model_artifact_uri = f"runs:/{best_run_id}/model_pipeline"

        logger.success(
            f"Best model found: {best_model_name} ({metric_to_optimize}: {best_metric_score:.4f})"
        )

        print("\n" + "=" * 30 + " MEJOR MODELO ENCONTRADO " + "=" * 30)
        print(f"Modelo:     {best_model_name}")
        print(f"Métrica:    {metric_to_optimize}")
        print(f"Valor:      {best_metric_score:.4f}")
        print(f"Run ID:     {best_run_id}")
        print(f"URI Modelo: {model_artifact_uri}")
        print("=" * 82)

        # Save best model locally
        try:
            logger.info("Loading and saving best model locally...")
            model = mlflow.sklearn.load_model(model_artifact_uri)

            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{best_model_name.lower()}_best_{timestamp}.pkl"
            model_path = MODELS_DIR / model_filename

            # Ensure models directory exists
            MODELS_DIR.mkdir(parents=True, exist_ok=True)

            # Save model
            joblib.dump(model, model_path)
            logger.success(f"Best model saved locally: {model_path}")
            print(f"\nModelo guardado localmente: {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model locally: {e}")
            print(f"\nAdvertencia: No se pudo guardar el modelo localmente: {e}")

        return {
            "model_name": best_model_name,
            "metric_name": metric_to_optimize,
            "score": best_metric_score,
            "run_id": best_run_id,
            "model_uri": model_artifact_uri,
        }
