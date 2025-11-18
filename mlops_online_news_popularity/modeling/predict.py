from pathlib import Path
from typing import List, Union

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import typer

from mlops_online_news_popularity.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


# =========================================================
# Prediction Function
# =========================================================


def predict(model: Union[Pipeline, object], X: pd.DataFrame) -> List[float]:
    """
    Make predictions using a trained model.

    Handles missing columns by filling with zeros to ensure compatibility
    with models trained on different feature sets.

    Parameters
    ----------
    model : Union[Pipeline, object]
        Trained sklearn pipeline or model with predict() method
    X : pd.DataFrame
        Input features

    Returns
    -------
    List[float]
        Predictions as list

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.linear_model import Ridge
    >>> model = Ridge()
    >>> X_train = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    >>> y_train = pd.Series([10, 20])
    >>> model.fit(X_train, y_train)
    >>> X_test = pd.DataFrame({'a': [5], 'b': [6]})
    >>> predictions = predict(model, X_test)
    >>> isinstance(predictions, list)
    True
    """
    # Get expected features from model
    expected_features = None

    # Try different ways to get feature names
    if hasattr(model, "feature_names_in_"):
        expected_features = model.feature_names_in_
    elif isinstance(model, Pipeline):
        # Try to get from first step in pipeline
        try:
            first_step = model.steps[0][1]
            if hasattr(first_step, "feature_names_in_"):
                expected_features = first_step.feature_names_in_
        except (IndexError, AttributeError):
            pass

    # If we still don't have features, use input columns
    if expected_features is None:
        expected_features = X.columns.tolist()

    # Add missing columns with zeros
    X_aligned = X.copy()
    for col in expected_features:
        if col not in X_aligned.columns:
            X_aligned[col] = 0

    # Reorder columns to match model (if we have expected features)
    if expected_features is not None:
        X_aligned = X_aligned[expected_features]

    # Make predictions
    predictions = model.predict(X_aligned)

    # Ensure we return a list
    if isinstance(predictions, np.ndarray):
        return predictions.tolist()
    elif isinstance(predictions, pd.Series):
        return predictions.tolist()
    else:
        return list(predictions)


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Performing inference for model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Inference complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
