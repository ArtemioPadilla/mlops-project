# Experiment Workflow

Workflow for conducting ML experiments with MLflow tracking.

## Experiment Setup

1. **Define hypothesis**: "Will RandomForest outperform Ridge?"
2. **Update configuration**: Add models to `config/models.yaml`
3. **Run experiment**: `make train`
4. **Analyze results**: MLflow UI

## Comparing Models

```bash
# 1. Configure models
cat > config/models.yaml << 'YAML'
experiment_name: "Model Comparison v1"
metric_to_optimize: "val_rmse"

models_to_try:
  Ridge:
    class_path: "sklearn.linear_model.Ridge"
  RandomForest:
    class_path: "sklearn.ensemble.RandomForestRegressor"
  XGBoost:
    class_path: "xgboost.XGBRegressor"
YAML

# 2. Run experiment
make train

# 3. View in MLflow
make mlflow-ui
```

## Hyperparameter Tuning

Add parameters to YAML:

```yaml
RandomForest:
  class_path: "sklearn.ensemble.RandomForestRegressor"
  n_estimators: 200
  max_depth: 10
  min_samples_split: 5
```

## Best Practices

- Use descriptive experiment names
- Document hypothesis in commit messages
- Track all experiments (even failures)
- Compare on validation set, report on test set
