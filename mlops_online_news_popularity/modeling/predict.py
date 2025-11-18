# mlops_online_news_popularity/modeling/predict.py

import joblib
import pandas as pd
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from mlops_online_news_popularity.config import MODELS_DIR, PROCESSED_DATA_DIR


# ============================================================
# REQUIRED BY PYTEST
# ============================================================

def load_model(model_path: str):
    """
    Minimal load_model function required by pytest.
    Loads a joblib-style model (usually a sklearn Pipeline).
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")

    logger.info(f"Loading model from: {model_path}")
    return joblib.load(model_path)



def predict(model, X):
    """
    Run prediction ensuring feature alignment with training columns
    and ALWAYS return a Python list (as required by tests).
    """
    logger.info("Running prediction...")

    # Ensure X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Ensure model has training column names
    if hasattr(model, "feature_names_in_"):
        train_cols = list(model.feature_names_in_)
    else:
        raise ValueError("Model does not contain feature_names_in_")

    # Add missing columns (fill with 0)
    for col in train_cols:
        if col not in X:
            X[col] = 0

    # Drop extra columns not seen during training
    X = X[train_cols]

    # Run prediction
    preds = model.predict(X)

    # Convert numpy array → list (tests expect list)
    return preds.tolist()




# ============================================================
# OPTIONAL CLI (your original content)
# ============================================================

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
):
    """
    Dummy CLI for inference. This is NOT used in pytest.
    """
    logger.info("Performing inference for model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Inference complete.")


if __name__ == "__main__":
    app()
