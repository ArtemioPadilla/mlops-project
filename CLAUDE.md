# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MLOps project for predicting Online News Popularity, built using the Cookiecutter Data Science template. The project uses Python 3.10 and follows a standard data science workflow with separate modules for data processing, feature engineering, and model training/inference.

## Development Commands

### Environment Setup
```bash
# Install the package in development mode (required for imports to work)
pip install -e .

# Install dependencies
make requirements

# Create virtual environment (if needed)
make create_environment
```

**Important**: Run `pip install -e .` first to install the package in editable mode. This allows Python to import `mlops_online_news_popularity` modules and prevents `ModuleNotFoundError`.

### Code Quality
```bash
# Run linting (flake8, isort, black)
make lint

# Auto-format code with black and isort
make format

# Clean compiled Python files
make clean
```

### Testing
```bash
# Run all tests
make test

# Run specific test file
python -m pytest tests/test_data.py

# Run specific test function
python -m pytest tests/test_data.py::test_function_name
```

### MLOps Pipeline

The project includes an end-to-end MLOps pipeline with preprocessing, training, and experiment tracking.

#### Preprocessing

Run the preprocessing pipeline to create train/val/test splits:

```bash
# Run preprocessing pipeline
make preprocess

# This creates:
# - data/processed/X_train.csv
# - data/processed/X_val.csv
# - data/processed/X_test.csv
# - data/processed/y_train.csv
# - data/processed/y_val.csv
# - data/processed/y_test.csv
# - data/processed/metadata.json

# Or run directly with options:
python -m mlops_online_news_popularity.cli.preprocess_cli --help
python -m mlops_online_news_popularity.cli.preprocess_cli \
  --input data/raw/online_news_modified.csv \
  --output-dir data/processed \
  --corr-threshold 0.9
```

#### Model Training

Train models with MLflow experiment tracking:

```bash
# Train and compare multiple models (recommended)
make train  # Uses config/models.yaml

# Or run directly:
python -m mlops_online_news_popularity.cli.train_cli train-compare config/models.yaml

# Train a single model for quick testing
make train-single  # Default: Ridge

# Train specific model
python -m mlops_online_news_popularity.cli.train_cli train-single --model ridge
python -m mlops_online_news_popularity.cli.train_cli train-single --model randomforest
python -m mlops_online_news_popularity.cli.train_cli train-single --model xgboost
```

#### Complete Pipeline

Run preprocessing + training (multiple models) in one command:

```bash
make pipeline  # Runs: preprocess + train (all models from config)
```

#### MLflow UI

View experiment results and compare models:

```bash
# Start MLflow UI (dev environment)
make mlflow-ui

# Or specify backend directly:
mlflow ui --backend-store-uri sqlite:///mlflow_artifacts/dev/mlflow.db --port 5001

# For quickstart environment:
mlflow ui --backend-store-uri sqlite:///mlflow_artifacts/quickstart/mlflow.db --port 5000
```

Then open http://localhost:5001 in your browser.

### Running Individual Modules

The main entry points for the MLOps pipeline are:

```bash
# Preprocessing pipeline (creates train/val/test splits)
python -m mlops_online_news_popularity.cli.preprocess_cli --help

# Model training with MLflow tracking
python -m mlops_online_news_popularity.cli.train_cli --help
```

For direct module usage:

```bash
# Model training (if calling directly)
python mlops_online_news_popularity/modeling/train.py

# Model inference/prediction
python mlops_online_news_popularity/modeling/predict.py
```

All CLI modules accept command-line arguments. Use `--help` to see available options.

## Code Architecture

### Directory Structure

The project follows Cookiecutter Data Science structure:

- **mlops_online_news_popularity/**: Main source package
  - **config.py**: Centralized configuration with path constants (DATA_DIR, MODELS_DIR, REPORTS_DIR, etc.) and loguru logger setup
  - **preprocessing/**: Unified data preprocessing module
    - **data_processor.py**: DataProcessor class - complete preprocessing pipeline (model-agnostic)
    - **data_cleaning.py**: DataCleaner class with comprehensive cleaning methods
    - **data_exploration.py**: DataExplorer class for EDA and profiling
    - **data_io.py**: DataLoader class for CSV operations
    - **data_comparison.py**: DataComparator class for dataset comparison
    - **utils.py**: Utility functions (e.g., classify_numeric_columns)
  - **modeling/**: Model training, evaluation, and comparison
    - **train.py**: ModelTrainer class with sklearn Pipeline integration
    - **compare.py**: Experimento class for MLflow-based model comparison
    - **predict.py**: Inference/prediction logic
  - **cli/**: Command-line interfaces for MLOps workflows
    - **preprocess_cli.py**: CLI for running preprocessing pipeline
    - **train_cli.py**: CLI for model training with MLflow tracking

- **data/**: Data directory (managed by DVC, excluded from git)
  - **raw/**: Original immutable data
  - **interim/**: Intermediate transformed data
  - **processed/**: Final datasets for modeling
  - **external/**: Third-party data sources

- **models/**: Trained models and model artifacts
- **notebooks/**: Jupyter notebooks for experimentation (numbered in workflow order)
  - `01_news_online_eda.ipynb` - Exploratory data analysis
  - `02_clean_preprocess.ipynb` - Initial data cleaning
  - `03_news_online_preprocess.ipynb` - Advanced preprocessing
  - `04_online_news_ml_models.ipynb` - Model training and evaluation
  - `05_mlflow_quickstart.ipynb` - MLflow tutorial and experiments
  - `06_testing_news_online_popularity.ipynb` - Model testing and validation
  - **archive/**: Old/deprecated notebook versions (e.g., `Limpieza_de_datos vo.ipynb`, `Limpieza_de_datos_v1.ipynb`)
- **tests/**: Test files using pytest
- **reports/**: Generated analysis outputs
  - **figures/**: Generated visualizations
- **docs/**: Generated data profiling reports (HTML)

### Module Architecture

The CLI modules (preprocess_cli.py, train_cli.py) and core modeling modules (train.py, predict.py) follow these conventions:
- Use typer for CLI argument parsing
- Import paths from `mlops_online_news_popularity.config`
- Use loguru for logging
- Use tqdm for progress bars
- Define commands with `@app.command()` decorators
- Can be run directly or imported as modules

### Key Configuration

- **Python Version**: 3.10
- **Code Style**: Black (line length 99), isort, flake8
- **Logging**: loguru integrated with tqdm
- **Data Versioning**: DVC is configured (data.dvc file present)
- **Package Management**: Uses pyproject.toml with flit as build backend

### Path Management

All paths should be imported from `mlops_online_news_popularity.config`:
- Use `RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, `INTERIM_DATA_DIR`, `EXTERNAL_DATA_DIR` for data paths
- Use `MODELS_DIR` for model artifacts
- Use `REPORTS_DIR` and `FIGURES_DIR` for outputs
- `PROJ_ROOT` points to the project root directory

This ensures consistent path handling across all modules.

### Code Style Rules

When editing code, respect these project conventions:
- Line length: 99 characters (enforced by black)
- Import sorting: Use isort with black profile
- Flake8 exceptions: E731, E266, E501, C901, W503 are ignored
- Excluded from linting: git, notebooks, references, models, data directories
- All scripts use typer for CLI interfaces
- All scripts use loguru for logging
- Use tqdm for long-running operations

### Working with Data

Data is tracked with DVC but the actual files are in `.gitignore`. When working with data:
- Never commit data files directly to git
- Data should be placed in the appropriate subdirectory (raw/interim/processed/external)
- Use the paths from config.py to reference data files

## Data Preprocessing Architecture

### Overview: Separation of Concerns

The project follows a clear separation between **model-agnostic** and **model-specific** preprocessing:

- **DataProcessor** (`preprocessing/data_processor.py`): Handles all model-agnostic steps (cleaning, feature engineering, splitting)
- **ModelTrainer** (`modeling/train.py`): Handles all model-specific transformations (scaling, imputation, power transforms)

This prevents data leakage and makes the pipeline modular and reusable.

### DataProcessor: Model-Agnostic Preprocessing

The `DataProcessor` orchestrates the complete preprocessing pipeline from raw data to clean train/val/test splits.

**Basic Usage**:
```python
from mlops_online_news_popularity.preprocessing import DataProcessor

# Initialize with filepath and configuration
processor = DataProcessor(
    filepath='Data/online_news_modified.csv',
    target_col='shares',
    cols_to_drop=['url', 'timedelta'],  # Non-predictive columns
    correlation_threshold=0.9  # Remove highly correlated features
)

# Execute complete preprocessing pipeline
processor.process()

# Access clean splits
X_train, y_train = processor.X_train, processor.y_train
X_val, y_val = processor.X_val, processor.y_val
X_test, y_test = processor.X_test, processor.y_test

# Access column classifications (for ModelTrainer)
binary_cols = processor.cols_bin
non_binary_cols = processor.cols_no_bin
```

**What DataProcessor Does**:
1. **Load and clean raw data** (using `DataCleaner` under the hood):
   - Clean primary key (URL validation)
   - Force numeric conversion
   - Apply business rules (timedelta clipping, LDA normalization)
2. **Feature engineering**:
   - Drop non-predictive columns
   - Classify columns as binary vs non-binary
3. **Split data** (70/15/15 train/val/test)
4. **Handle high correlation** (on train set only, prevents data leakage)

**Output**: Clean, split data ready for model training + metadata about column types

### DataCleaner: Low-Level Cleaning Utilities

For custom cleaning workflows, use `DataCleaner` directly:

```python
from mlops_online_news_popularity.preprocessing import DataCleaner

cleaner = DataCleaner(df)
cleaned_df = (cleaner
    .clean_primary_key(key="url")
    .force_numeric(exclude=["url"])
    .apply_business_rules()
    .normalize_lda(["LDA_00", "LDA_01", "LDA_02", "LDA_03", "LDA_04"])
    .get_df())
```

### DataExplorer: EDA and Profiling

```python
from mlops_online_news_popularity.preprocessing import DataExplorer

# Basic EDA
DataExplorer.explore_data(df)

# Correlation heatmap
DataExplorer.plot_correlation_matrix(df, title="Correlation Matrix", save_path="docs/corr.png")

# Profiling report
DataExplorer.generate_profiling_report(df, title="Data Report", output_dir="docs", filename="report.html")
```

### Other Utilities

```python
from mlops_online_news_popularity.preprocessing import DataLoader, DataComparator, classify_numeric_columns

# Load/save CSV
loader = DataLoader()
df = loader.load_csv("data/raw/dataset.csv")

# Compare datasets
comparator = DataComparator(original_df, cleaned_df)
report = (comparator
    .compare_stats()
    .add_differences()
    .export_report("reports/comparison.csv"))

# Classify column types
binary_cols, non_binary_cols = classify_numeric_columns(df)
```

## Model Training with ModelTrainer

### Overview: Model-Specific Transformations

`ModelTrainer` receives clean data from `DataProcessor` and applies **model-specific** transformations:
- Imputation strategies
- Power transformations (for skewed distributions)
- Scaling (StandardScaler for non-binary features)
- Target transformation (log transform for skewed targets)

All transformations are done within a sklearn Pipeline to prevent data leakage (fitted only on train set).

### Basic Workflow

```python
from mlops_online_news_popularity.preprocessing import DataProcessor
from mlops_online_news_popularity.modeling.train import ModelTrainer
from sklearn.ensemble import RandomForestRegressor

# Step 1: Model-agnostic preprocessing
processor = DataProcessor(filepath='Data/online_news_modified.csv')
processor.process()

# Step 2: Model-specific training
trainer = ModelTrainer(
    data_processor=processor,
    estimator=RandomForestRegressor(n_estimators=100, random_state=42),
    model_name="Random Forest"
)

# Step 3: Transform target (optional, for skewed distributions)
trainer.transform_target(apply_log=True)

# Step 4: Train and evaluate
trainer.train_model()
metrics = trainer.evaluate_model()

# Step 5: Cross-validation (optional)
cv_results = trainer.cross_validate_model(cv=5)
```

### What ModelTrainer Does

1. **Builds preprocessing pipeline** (fitted only on train):
   - **Non-binary columns**: Impute (median) → PowerTransform → StandardScale
   - **Binary columns**: Impute (most_frequent) only
2. **Transforms target** (optional): log(1 + y) for skewed distributions
3. **Calculates baseline RMSE**: Standard deviation of target (underfitting detection)
4. **Trains model**: Fits sklearn Pipeline (preprocessing + model) on train set
5. **Evaluates**: Computes metrics on train/val/test with diagnostic checks

### Evaluation Metrics

The `evaluate_model()` method provides comprehensive diagnostics:

```python
metrics = trainer.evaluate_model()
# Output:
# ==============================================================
# EVALUATION RESULTS - Random Forest
# ==============================================================
# Baseline RMSE: 1.2345
#
# Split      RMSE         MAE          R²           Status
# --------------------------------------------------------------
# TRAIN      0.9234       0.6789       0.8456       ✓ OK
# VAL        1.0123       0.7234       0.7890
# TEST       1.0089       0.7198       0.7912
#
# Gap (train - val RMSE): -0.0889  ✓ Good generalization
# ==============================================================
```

**Diagnostic Checks**:
- **Underfitting**: Train RMSE > Baseline RMSE
- **Overfitting**: Large negative gap between train and val RMSE
- **Good fit**: |Gap| < 0.05

### Model Comparison with MLflow

The `Experimento` class orchestrates multi-model experiments.

**Complete Example**:
```python
from mlops_online_news_popularity.preprocessing import DataProcessor
from mlops_online_news_popularity.modeling.compare import Experimento

# Step 1: Preprocess data (once for all models)
processor = DataProcessor(filepath='Data/online_news_modified.csv')
processor.process()

# Step 2: Run comparison experiment
experiment = Experimento(
    config_path="Data/config.yml",
    data_processor=processor
)
experiment.ejecuta_experimentos()

# Step 3: Get best model
best_model_info = experiment.mejor_modelo()
# Returns: {model_name, metric_name, score, run_id, model_uri}
```

**YAML Configuration** (`Data/config.yml`):
```yaml
experiment_name: "News Popularity Model Comparison"
metric_to_optimize: "val_rmse"
optimize_mode: "ASC"  # ASC for RMSE/MAE, DESC for R²

models_to_try:
  Ridge:
    class_path: "sklearn.linear_model.Ridge"
  RandomForest:
    class_path: "sklearn.ensemble.RandomForestRegressor"
  GradientBoosting:
    class_path: "sklearn.ensemble.GradientBoostingRegressor"
```

**Note**: Scaling is handled automatically by `ModelTrainer`. No need to specify `scaler` in YAML.

## MLflow Experiment Tracking

The project uses MLflow for experiment tracking with a two-environment setup.

### Two MLflow Environments

**Quickstart Environment** (`mlflow_artifacts/quickstart/`):
- For learning, tutorials, and experimentation
- Fully gitignored and safe to delete/reset anytime
- Use `MLFLOW_QUICKSTART_URI` from config.py
- See `notebooks/mlflow-quickstart.ipynb` for tutorial

**Dev Environment** (`mlflow_artifacts/dev/`):
- For actual model development and training
- Databases and artifacts are gitignored, but directory structure is tracked
- Use `MLFLOW_DEV_URI` or `MLFLOW_TRACKING_URI` (default) from config.py
- Persists across sessions - contains your real experiment history

### Using MLflow in Code

```python
from mlops_online_news_popularity.config import MLFLOW_TRACKING_URI
import mlflow

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)  # Uses dev environment by default
mlflow.set_experiment("experiment-name")

with mlflow.start_run():
    mlflow.log_param("param_name", value)
    mlflow.log_metric("metric_name", value)
    # ... training code ...
```

### Viewing MLflow UI

```bash
# View quickstart experiments
mlflow ui --backend-store-uri sqlite:///mlflow_artifacts/quickstart/mlflow.db --port 5000

# View dev experiments (default)
mlflow ui --backend-store-uri sqlite:///mlflow_artifacts/dev/mlflow.db --port 5001

# Or use the shorthand (from project root)
cd mlops-project
mlflow ui  # Uses default tracking URI
```

Then open http://localhost:5000 in your browser.

### Reset Quickstart Environment

```bash
# Safe to run anytime - only affects quickstart environment
rm -rf mlops-project/mlflow_artifacts/quickstart/
```

### Environment Variable Override

Create a `.env` file (from `.env.example`) to override the default tracking URI:

```bash
# .env
MLFLOW_TRACKING_URI=sqlite:///mlflow_artifacts/quickstart/mlflow.db
# or for remote server:
# MLFLOW_TRACKING_URI=http://mlflow-server:5000
```
