# API Reference Overview

Auto-generated API documentation from Python docstrings using mkdocstrings.

## Quick Navigation

| Module | Description |
|--------|-------------|
| [Preprocessing](preprocessing.md) | Data cleaning, processing, and splitting |
| [Modeling](modeling.md) | Model training and experiment tracking |
| [CLI](cli.md) | Command-line interfaces |

## Module Structure

```
mlops_online_news_popularity/
├── preprocessing/
│   ├── DataProcessor       # Main preprocessing orchestrator
│   ├── DataCleaner         # Low-level cleaning utilities
│   ├── DataExplorer        # EDA and profiling
│   ├── DataLoader          # CSV I/O operations
│   ├── DataComparator      # Dataset comparison
│   └── utils               # Helper functions
├── modeling/
│   ├── ModelTrainer        # Model-specific training
│   └── Experimento         # Multi-model comparison
└── cli/
    ├── preprocess_cli      # Preprocessing commands
    └── train_cli           # Training commands
```

## Usage Examples

### Import Classes

```python
from mlops_online_news_popularity.preprocessing import DataProcessor
from mlops_online_news_popularity.modeling.train import ModelTrainer
from mlops_online_news_popularity.modeling.compare import Experimento
```

### Basic Workflow

```python
# 1. Preprocess data
processor = DataProcessor(filepath='data/raw/online_news_modified.csv')
processor.process()

# 2. Train model
from sklearn.ensemble import RandomForestRegressor

trainer = ModelTrainer(
    data_processor=processor,
    estimator=RandomForestRegressor(random_state=42),
    model_name="Random Forest"
)

# Optional: Transform target
trainer.transform_target(apply_log=True)

# Train and evaluate
trainer.train_model()
metrics = trainer.evaluate_model()

# 3. Multi-model comparison
experiment = Experimento(
    config_path="config/models.yaml",
    data_processor=processor
)
experiment.ejecuta_experimentos()
best_model = experiment.mejor_modelo()
```

## Documentation Features

All API documentation is **auto-generated** from source code docstrings, which means:

- ✅ **Always up-to-date**: Documentation updates automatically when code changes
- ✅ **Type hints rendered**: Function signatures show types
- ✅ **Source links**: Direct links to GitHub source code
- ✅ **Searchable**: Integrated with MkDocs search
- ✅ **Cross-referenced**: Automatic links between classes/functions

## Docstring Style

This project uses **NumPy-style** docstrings with comprehensive sections:

- **Parameters**: Function/method parameters with types
- **Returns**: Return values and types
- **Raises**: Exceptions that may be raised
- **Examples**: Usage examples
- **Notes**: Additional information

See individual module pages for complete API documentation with full docstrings.
