# Models YAML Configuration

Configure models for training in `config/models.yaml`.

## Structure

```yaml
# Experiment settings
experiment_name: "News Popularity Prediction"
metric_to_optimize: "val_rmse"  # Metric to select best model
optimize_mode: "ASC"  # ASC for RMSE/MAE, DESC for R²

# Models to train
models_to_try:
  Ridge:
    class_path: "sklearn.linear_model.Ridge"
    # Optional: Add model parameters
    alpha: 1.0

  RandomForest:
    class_path: "sklearn.ensemble.RandomForestRegressor"
    n_estimators: 100
    max_depth: 10
    random_state: 42

  XGBoost:
    class_path: "xgboost.XGBRegressor"
    n_estimators: 100
    learning_rate: 0.1
```

## Adding New Models

1. Add model to `models_to_try`
2. Specify `class_path` (full Python import path)
3. Add parameters (optional)
4. Run `make train`

## Available Metrics

- `val_rmse`: Validation RMSE
- `val_mae`: Validation MAE
- `val_r2`: Validation R²
- `test_rmse`: Test RMSE
- `test_mae`: Test MAE
- `test_r2`: Test R²
