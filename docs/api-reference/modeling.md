# Modeling API Reference

API reference for modeling modules.

## ModelTrainer

**Class**: `mlops_online_news_popularity.modeling.train.ModelTrainer`

### Constructor

```python
ModelTrainer(
    data_processor: DataProcessor,
    estimator: sklearn estimator,
    model_name: str
)
```

### Methods

#### `transform_target(apply_log: bool = True) -> None`
Apply log transformation to target.

#### `train_model() -> None`
Train the sklearn Pipeline.

#### `evaluate_model() -> dict`
Evaluate on train/val/test sets.

Returns: Dictionary with metrics for each split.

#### `cross_validate_model(cv: int = 5) -> dict`
Perform cross-validation.

## Experimento

**Class**: `mlops_online_news_popularity.modeling.compare.Experimento`

### Constructor

```python
Experimento(
    config_path: str,
    data_processor: DataProcessor
)
```

### Methods

#### `ejecuta_experimentos() -> None`
Train all models and log to MLflow.

#### `mejor_modelo() -> dict`
Return best model information.
