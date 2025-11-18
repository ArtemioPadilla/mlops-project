# mlops_online_news_popularity/modeling/train.py

import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train, output_dir: Path):
    """
    Minimal trainer required by pytest test_full_pipeline.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pkl"

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), X_train.columns.tolist()),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, model_path)

    return str(model_path)
