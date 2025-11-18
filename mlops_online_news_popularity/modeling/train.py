# model_trainer.py
# -*- coding: utf-8 -*-
"""
Model training module with comprehensive pipeline support.

This module provides the ModelTrainer class for training regression models
with automated model-specific preprocessing, evaluation, and cross-validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
import typer

from mlops_online_news_popularity.config import MODELS_DIR, PROCESSED_DATA_DIR
from mlops_online_news_popularity.preprocessing import DataProcessor
from mlops_online_news_popularity.preprocessing.utils import classify_numeric_columns

app = typer.Typer()


# =========================================================
# ModelTrainer
# =========================================================


class ModelTrainer:
    """
    Model trainer with automated preprocessing pipeline.

    This class handles MODEL-SPECIFIC transformations and training.
    It receives clean train/val/test splits from DataProcessor and applies
    transformations that depend on the model type (imputation, scaling, etc.).

    Workflow:
    ---------
    1. Receive data from DataProcessor (already cleaned and split)
    2. Build preprocessing pipeline (imputation, power transform, scaling)
    3. Optionally transform target (log transformation for skewed distributions)
    4. Train model (fit pipeline on train set)
    5. Evaluate on train/val/test with comprehensive metrics

    Example:
    --------
    >>> from mlops_online_news_popularity.preprocessing import DataProcessor
    >>> from sklearn.ensemble import RandomForestRegressor
    >>>
    >>> # Step 1: Model-agnostic preprocessing
    >>> processor = DataProcessor(filepath='data/raw/data.csv')
    >>> processor.process()
    >>>
    >>> # Step 2: Model-specific training
    >>> trainer = ModelTrainer(
    ...     data_processor=processor,
    ...     estimator=RandomForestRegressor(random_state=42),
    ...     model_name="Random Forest"
    ... )
    >>> trainer.transform_target(apply_log=True)
    >>> trainer.train_model()
    >>> metrics = trainer.evaluate_model()
    """

    def __init__(
        self,
        data_processor: DataProcessor,
        estimator: BaseEstimator,
        model_name: Optional[str] = None,
        fit_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize ModelTrainer.

        Parameters
        ----------
        data_processor : DataProcessor
            Processed data with train/val/test splits and column metadata
        estimator : BaseEstimator
            Sklearn estimator to train
        model_name : str, optional
            Human-readable model name (default: estimator class name)
        fit_params : Dict[str, Any], optional
            Additional parameters for model.fit() (e.g., eval_set for XGBoost)
        """
        # Extract data from DataProcessor
        self.X_train = data_processor.X_train
        self.X_val = data_processor.X_val
        self.X_test = data_processor.X_test
        self.y_train = data_processor.y_train.copy()  # Copy for potential transformation
        self.y_val = data_processor.y_val.copy()
        self.y_test = data_processor.y_test.copy()

        # Extract column classifications
        self.cols_bin = data_processor.cols_bin
        self.cols_no_bin = data_processor.cols_no_bin

        # Model configuration
        self.estimator = estimator
        self.model_name = model_name or estimator.__class__.__name__
        self.fit_params = fit_params or {}

        # Build preprocessing pipeline
        preprocessor = self._build_preprocessor()
        self.pipeline = self._create_pipeline(estimator, preprocessor)

        # State tracking
        self._is_fitted: bool = False
        self._target_transformed: bool = False
        self.baseline_rmse: Optional[float] = None

    # --------------------------
    # Preprocessing Pipeline
    # --------------------------

    def _build_preprocessor(self) -> ColumnTransformer:
        """
        Build model-specific preprocessing pipeline.

        Strategy:
        ---------
        - NON-BINARY columns: Impute (median) → PowerTransform → StandardScale
        - BINARY columns: Impute (most_frequent) only

        Returns
        -------
        ColumnTransformer
            Preprocessing transformations
        """
        # NON-BINARY: Full preprocessing pipeline
        numeric_non_binary_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("power", PowerTransformer(method="yeo-johnson")),
                ("scaler", StandardScaler()),
            ]
        )

        # BINARY: Only imputation
        binary_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])

        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ("num_non_bin", numeric_non_binary_transformer, self.cols_no_bin),
                ("num_bin", binary_transformer, self.cols_bin),
            ],
            remainder="passthrough",
        )

        return preprocessor

    def _create_pipeline(
        self, estimator: BaseEstimator, preprocessor: ColumnTransformer
    ) -> Pipeline:
        """
        Create complete pipeline: preprocessing + model.

        Parameters
        ----------
        estimator : BaseEstimator
            The model
        preprocessor : ColumnTransformer
            Preprocessing steps

        Returns
        -------
        Pipeline
            Complete pipeline
        """
        steps = [("preprocessor", preprocessor), ("model", estimator)]
        return Pipeline(steps)

    # --------------------------
    # Target Transformation
    # --------------------------

    def transform_target(self, apply_log: bool = True) -> "ModelTrainer":
        """
        Transform target variable (optional).

        For highly skewed targets (like 'shares'), applying log(1 + y)
        can improve model performance.

        Parameters
        ----------
        apply_log : bool, optional
            Apply log(1 + y) transformation (default: True)

        Returns
        -------
        ModelTrainer
            Self for method chaining
        """
        if self._target_transformed:
            logger.warning("Target already transformed, skipping")
            return self

        if apply_log:
            logger.info("Applying log(1 + y) transformation to target")
            self.y_train = np.log1p(self.y_train)
            self.y_val = np.log1p(self.y_val)
            self.y_test = np.log1p(self.y_test)

            # Calculate baseline RMSE (standard deviation of log-transformed target)
            self.baseline_rmse = float(self.y_train.std())
            logger.info(f"Baseline RMSE (std of y_train_log): {self.baseline_rmse:.4f}")
            logger.info(f"⚠️  Models with RMSE > {self.baseline_rmse:.4f} are UNDERFITTING")

            self._target_transformed = True

        return self

    # --------------------------
    # Training
    # --------------------------

    def train_model(self) -> "ModelTrainer":
        """
        Train the model pipeline.

        Fits both preprocessing steps and the model on the training set.

        Returns
        -------
        ModelTrainer
            Self for method chaining
        """
        logger.info(f"Training model: {self.model_name}")
        self.pipeline.fit(self.X_train, self.y_train, **self.fit_params)
        self._is_fitted = True
        logger.success(f"Model {self.model_name} trained successfully")
        return self

    # --------------------------
    # Evaluation
    # --------------------------

    @staticmethod
    def _regression_metrics(
        y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate regression metrics.

        Parameters
        ----------
        y_true : Union[pd.Series, np.ndarray]
            True values
        y_pred : Union[pd.Series, np.ndarray]
            Predicted values

        Returns
        -------
        Dict[str, float]
            Dictionary with rmse, mae, r2
        """
        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}

    def evaluate_model(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model on train/val/test sets.

        Computes metrics and diagnostic information:
        - RMSE, MAE, R² for each split
        - Train-val and val-test gaps
        - Underfitting detection (if baseline_rmse is available)

        Returns
        -------
        Dict[str, Dict[str, float]]
            Nested dictionary with metrics per split and gaps
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be trained before evaluation (call train_model)")

        logger.info("Evaluating model on train/val/test sets")

        # Predictions
        yhat_train = self.pipeline.predict(self.X_train)
        yhat_val = self.pipeline.predict(self.X_val)
        yhat_test = self.pipeline.predict(self.X_test)

        # Metrics
        metrics = {
            "train": self._regression_metrics(self.y_train, yhat_train),
            "val": self._regression_metrics(self.y_val, yhat_val),
            "test": self._regression_metrics(self.y_test, yhat_test),
        }

        # Diagnostic gaps
        metrics["gaps"] = {
            "rmse_train_val": metrics["train"]["rmse"] - metrics["val"]["rmse"],
            "rmse_val_test": metrics["val"]["rmse"] - metrics["test"]["rmse"],
        }

        # Pretty print
        self._pretty_print_metrics(metrics)

        return metrics

    def _pretty_print_metrics(self, metrics: Dict[str, Dict[str, float]]) -> None:
        """
        Pretty print evaluation metrics.

        Parameters
        ----------
        metrics : Dict[str, Dict[str, float]]
            Metrics dictionary from evaluate_model()
        """
        name = self.model_name
        print("\n" + "=" * 70)
        print(f"EVALUATION RESULTS - {name}")
        print("=" * 70)

        # Show baseline if available
        if self.baseline_rmse is not None:
            print(f"Baseline RMSE: {self.baseline_rmse:.4f}\n")

        # Header
        print(f"{'Split':<10} {'RMSE':<12} {'MAE':<12} {'R²':<12} {'Status':<20}")
        print("-" * 70)

        # Metrics per split
        for split in ("train", "val", "test"):
            m = metrics[split]
            status = ""

            # Underfitting check (train vs baseline)
            if self.baseline_rmse and split == "train":
                if m["rmse"] > self.baseline_rmse:
                    status = "⚠️ UNDERFITTING"
                else:
                    status = "✓ OK"

            print(
                f"{split.upper():<10} {m['rmse']:<12.4f} {m['mae']:<12.4f} "
                f"{m['r2']:<12.4f} {status:<20}"
            )

        # Overfitting check (train-val gap)
        gap = metrics["gaps"]["rmse_train_val"]
        print(f"\nGap (train - val RMSE): {gap:.4f}", end="")

        if gap < -0.05:
            print("  ⚠️ Possible OVERFITTING")
        elif abs(gap) <= 0.05:
            print("  ✓ Good generalization")
        else:
            print("  ⚠️ Review model")

        print("=" * 70)

    # --------------------------
    # Cross-Validation
    # --------------------------

    def cross_validate_model(
        self,
        cv: int = 5,
        scoring: Optional[Union[str, Iterable[str], Dict[str, Any]]] = None,
        n_jobs: Optional[int] = None,
        return_train_score: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation.

        Parameters
        ----------
        cv : int, optional
            Number of folds (default: 5)
        scoring : Optional[Union[str, Iterable[str], Dict[str, Any]]], optional
            Scoring metrics (default: rmse, mae, r2)
        n_jobs : Optional[int], optional
            Number of parallel jobs (default: None)
        return_train_score : bool, optional
            Return train scores (default: True)

        Returns
        -------
        Dict[str, Any]
            Dictionary with 'raw' (sklearn output) and 'summary' (aggregated metrics)
        """
        if scoring is None:
            scoring = {
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "r2": "r2",
            }

        logger.info(f"Running {cv}-fold cross-validation")

        cv_results = cross_validate(
            estimator=self.pipeline,
            X=self.X_train,
            y=self.y_train,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            return_train_score=return_train_score,
        )

        # Aggregate results
        summary: Dict[str, Dict[str, float]] = {}
        for key, values in cv_results.items():
            if key.startswith("test_") or key.startswith("train_"):
                metric = key.split("_", 1)[1]
                vals = np.array(values, dtype=float)

                # sklearn uses negative values for loss metrics
                if metric in ("rmse", "mae"):
                    vals = -vals

                summary[key] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                }

        # Pretty print
        self._pretty_print_cv(summary, cv)

        return {"raw": cv_results, "summary": summary}

    def _pretty_print_cv(self, summary: Dict[str, Dict[str, float]], cv: int) -> None:
        """
        Pretty print cross-validation results.

        Parameters
        ----------
        summary : Dict[str, Dict[str, float]]
            Summary from cross_validate_model()
        cv : int
            Number of folds
        """
        print("\n" + "=" * 70)
        print(f"CROSS-VALIDATION (cv={cv}) - {self.model_name}")
        print("=" * 70)

        for split in ("train", "test"):
            for metric in ("rmse", "mae", "r2"):
                key = f"{split}_{metric}"
                if key in summary:
                    m = summary[key]
                    print(f"{key:12s} | mean: {m['mean']:.4f} | std: {m['std']:.4f}")

        print("=" * 70)


# =========================================================
# Functional API (Convenience Wrapper)
# =========================================================


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    estimator: Optional[BaseEstimator] = None,
    apply_log_transform: bool = True,
) -> Pipeline:
    """
    Functional wrapper for quick model training.

    This is a convenience function for simple use cases and testing.
    For full control and advanced features, use the ModelTrainer class directly.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_val : pd.DataFrame, optional
        Validation features (if None, uses first row of X_train)
    y_val : pd.Series, optional
        Validation target (if None, uses first value of y_train)
    X_test : pd.DataFrame, optional
        Test features (if None, uses first row of X_train)
    y_test : pd.Series, optional
        Test target (if None, uses first value of y_train)
    estimator : BaseEstimator, optional
        Sklearn estimator to train (default: Ridge with random_state=42)
    apply_log_transform : bool, optional
        Apply log(1 + y) transformation to target (default: True)

    Returns
    -------
    Pipeline
        Trained sklearn pipeline (preprocessing + model)

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> X = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]})
    >>> y = pd.Series([10, 20, 30, 40])
    >>> model = train_model(X, y, estimator=RandomForestRegressor(n_estimators=10))
    >>> predictions = model.predict(X)
    >>> len(predictions) == len(y)
    True

    Notes
    -----
    This function creates a minimal DataProcessor-compatible object internally
    and delegates to ModelTrainer for the actual training.
    """
    from sklearn.linear_model import Ridge

    # Create minimal DataProcessor-like object for ModelTrainer compatibility
    class MinimalDataProcessor:
        """Minimal data processor wrapper for functional API."""

        def __init__(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: Optional[pd.DataFrame],
            y_val: Optional[pd.Series],
            X_test: Optional[pd.DataFrame],
            y_test: Optional[pd.Series],
        ):
            self.X_train = X_train
            self.y_train = y_train

            # Use dummy data if validation/test sets not provided
            self.X_val = X_val if X_val is not None else X_train.head(1).copy()
            self.y_val = y_val if y_val is not None else y_train.head(1).copy()
            self.X_test = X_test if X_test is not None else X_train.head(1).copy()
            self.y_test = y_test if y_test is not None else y_train.head(1).copy()

            # Classify columns as binary vs non-binary
            binary_cols, non_binary_cols = classify_numeric_columns(X_train)
            self.cols_bin = binary_cols
            self.cols_no_bin = non_binary_cols

    # Use Ridge as default estimator
    if estimator is None:
        estimator = Ridge(random_state=42)

    # Create minimal data processor
    data_processor = MinimalDataProcessor(X_train, y_train, X_val, y_val, X_test, y_test)

    # Train using ModelTrainer
    trainer = ModelTrainer(data_processor=data_processor, estimator=estimator)

    if apply_log_transform:
        trainer.transform_target(apply_log=True)

    trainer.train_model()

    return trainer.pipeline


# =========================================================
# CLI Interface
# =========================================================


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    """
    Model training CLI.

    This is a template - customize for your specific use case.
    For full functionality, use ModelTrainer programmatically:

        from mlops_online_news_popularity.preprocessing import DataProcessor
        from mlops_online_news_popularity.modeling.train import ModelTrainer
    """
    logger.info("Model training module loaded")
    logger.info("For usage examples, see class docstring of ModelTrainer")


if __name__ == "__main__":
    app()
