"""
Model comparison module for MLflow experiment orchestration.

This module provides the Experimento class for comparing multiple regression models
using MLflow for experiment tracking and model comparison.
"""

import importlib
import inspect
from typing import Any, Dict

from loguru import logger
import yaml

import mlflow
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
#           config_path="data/config.yaml",
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

    def __init__(self, config_path: str, data_processor: DataProcessor):
        """
        Initialize the experiment with configuration and data.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file
        data_processor : DataProcessor
            DataProcessor object containing train/val/test splits
        """
        self.config_path = config_path
        self.data = data_processor

        # Import the YAML configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.experiment_name = self.config["experiment_name"]
        self.models_config = self.config["models_to_try"]

        # Initialize MLflow
        mlflow.set_experiment(self.experiment_name)
        logger.info(f"MLflow: Experiment to run '{self.experiment_name}'")
        print(f"MLflow: Experimento a Correr '{self.experiment_name}'")

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
                init_kwargs["random_state"] = 42

            return EstimatorClass(**init_kwargs)
        except Exception as e:
            logger.error(f"Error de instancia {class_path}: {e}")
            print(f"Error de instancia {class_path}: {e}")
            raise

    def ejecuta_experimentos(self) -> None:
        """
        Run all models specified in config.yaml and compare them.

        This creates a parent run in MLflow and nested child runs for each model.
        Each model is trained, evaluated, and logged to MLflow.
        """

        # Create the parent process for MLFLOW which contains the entire code execution

        with mlflow.start_run(run_name="Comparacion de modelos ") as parent_run:
            mlflow.log_param("config_file", self.config_path)

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
                        mlflow.set_tag("model_name", model_name)
                        mlflow.log_param("class_path", model_cfg["class_path"])
                        mlflow.log_param("target_transform", "log")  # Always log transform

                        # Creating ModelTrainer object to train and evaluate models
                        trainer = ModelTrainer(
                            data_processor=self.data,
                            estimator=estimator,
                            model_name=model_name,
                        )

                        # Transform target if needed (log transformation for skewed data)
                        trainer.transform_target(apply_log=True)

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
    # config.yaml previously saved in MLFLOW
    # -------------------------------------------------------------------------
    def mejor_modelo(self) -> Dict[str, Any]:
        """
        Connect with MLFlow to review which model is considered the best in the
        experiment run from config.yaml.

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

        metric_to_optimize = self.config["metric_to_optimize"].replace("_", ".")
        order_mode = self.config["optimize_mode"]

        # Get the experiment ID to evaluate
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not experiment:
            logger.error("No experiment ID found, check saved data or ID")
            print("No se encontro el ID, revisar datos guardados o el ID")
            return {}

        # Search for experiment results
        best_run_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_to_optimize} {order_mode}"],
            max_results=1,  # search for the best model
        )

        if best_run_df.empty:
            logger.warning("No experiments found")
            print("No se Encontraron Experimentos ejecutados")
            return {}

        # Export the information of the best result
        best_run_data = best_run_df.iloc[0]
        best_model_name = best_run_data["tags.model_name"]
        best_run_id = best_run_data["run_id"]
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

        return {
            "model_name": best_model_name,
            "metric_name": metric_to_optimize,
            "score": best_metric_score,
            "run_id": best_run_id,
            "model_uri": model_artifact_uri,
        }
