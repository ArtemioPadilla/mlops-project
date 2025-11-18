import os
import joblib
from sklearn.ensemble import RandomForestRegressor

def train_model(X_train, y_train, output_dir=None):
    """
    Train a RandomForest model. Output_dir is optional for unit tests.
    """

    # 1) Allow training without saving (for pytest)
    if output_dir is None:
        output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    model_path = os.path.join(output_dir, "model.pkl")
    joblib.dump(model, model_path)

    return model
