# Experiment Tracking with MLflow

Multi-model comparison using MLflow.

## Experimento Class

Orchestrates training of multiple models and tracks experiments in MLflow.

## Usage

```python
from mlops_online_news_popularity.modeling.compare import Experimento

experiment = Experimento(
    config_path="config/models.yaml",
    data_processor=processor
)

# Run all experiments
experiment.ejecuta_experimentos()

# Get best model
best = experiment.mejor_modelo()
```

## MLflow Hierarchy

```mermaid
graph TB
    Exp[Experiment:<br/>"Impacto de Publicacion"]
    Parent[Parent Run]
    Ridge[Child: Ridge]
    RF[Child: RandomForest]
    KNN[Child: KNeighbors]

    Exp --> Parent
    Parent --> Ridge
    Parent --> RF
    Parent --> KNN

    style Parent fill:#f1faee
    style Ridge fill:#a8dadc
    style RF fill:#a8dadc
    style KNN fill:#a8dadc
```

## Configuration

```yaml
# config/models.yaml
experiment_name: "News Popularity"
metric_to_optimize: "val_rmse"
optimize_mode: "ASC"

models_to_try:
  Ridge:
    class_path: "sklearn.linear_model.Ridge"
  RandomForest:
    class_path: "sklearn.ensemble.RandomForestRegressor"
```

See [Models YAML Config](../configuration/models-yaml.md).
