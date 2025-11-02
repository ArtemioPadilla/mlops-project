# Model Selection

Strategies for selecting the best model.

## Selection Criteria

The `Experimento` class selects the best model based on a specified metric.

## Metrics Available

- **val_rmse**: Validation RMSE (lower is better)
- **val_mae**: Validation MAE (lower is better)
- **val_r2**: Validation R² (higher is better)

## Configuration

```yaml
metric_to_optimize: "val_rmse"
optimize_mode: "ASC"  # ASC for RMSE/MAE, DESC for R²
```

## Best Model Retrieval

```python
best_model_info = experiment.mejor_modelo()
# Returns:
# {
#   'model_name': 'RandomForest',
#   'metric_name': 'val_rmse',
#   'score': 0.8234,
#   'run_id': '...',
#   'model_uri': 'runs:/...'
# }
```

The best model is automatically saved to `models/` directory.
