# MLOps: Online News Popularity Prediction

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A complete MLOps pipeline for predicting the popularity (shares) of online news articles using machine learning. Built with scikit-learn, MLflow, and DVC following best practices for reproducible ML.

## Features

- **Automated Data Pipeline**: End-to-end preprocessing with cleaning, feature engineering, and train/val/test splitting
- **MLflow Experiment Tracking**: Compare multiple models with automated metric logging and artifact versioning
- **DVC Data Versioning**: Track and version datasets for reproducibility
- **Modular Architecture**: Clean separation between model-agnostic preprocessing and model-specific transformations
- **CLI Interfaces**: Simple command-line tools for all major operations
- **Multiple Models**: Support for Ridge, RandomForest, KNeighbors, XGBoost, and more
- **Comprehensive Evaluation**: RMSE, MAE, R² metrics with overfitting/underfitting diagnostics

## Quick Start

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/ArtemioPadilla/mlops-project.git
cd mlops-project

# 2. Create virtual environment
make create_environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install package in development mode (required!)
pip install -e .

# 4. Install dependencies
make requirements
```

### Run the Pipeline

```bash
# Run full pipeline (preprocessing + training)
make pipeline

# Or run steps individually:
make preprocess  # Create train/val/test splits
make train       # Train all models from config/models.yaml

# View results in MLflow UI
make mlflow-ui   # Open http://localhost:5001
```

## Available Commands

| Command | Description |
|---------|-------------|
| `make requirements` | Install dependencies |
| `make preprocess` | Run preprocessing pipeline |
| `make train` | Train all models from config |
| `make train-single` | Train a single model (default: Ridge) |
| `make pipeline` | Run full pipeline (preprocess + train) |
| `make mlflow-ui` | Start MLflow UI (port 5001) |
| `make docs` | Build documentation |
| `make docs-serve` | Serve documentation locally (port 8000) |
| `make docs-deploy` | Deploy documentation to GitHub Pages |
| `make lint` | Run code quality checks (flake8, isort, black) |
| `make format` | Auto-format code |
| `make test` | Run pytest tests |
| `make clean` | Remove compiled Python files |
| `make help` | Show all available commands |

## CLI Usage

### Preprocessing

```bash
# Run with default settings
python -m mlops_online_news_popularity.cli.preprocess_cli

# Custom configuration
python -m mlops_online_news_popularity.cli.preprocess_cli \
  --input data/raw/online_news_modified.csv \
  --output-dir data/processed \
  --corr-threshold 0.9
```

### Model Training

```bash
# Train and compare multiple models
python -m mlops_online_news_popularity.cli.train_cli train-compare config/models.yaml

# Train a single model
python -m mlops_online_news_popularity.cli.train_cli train-single --model ridge
python -m mlops_online_news_popularity.cli.train_cli train-single --model randomforest
```

## MLflow Tracking

The project includes two MLflow environments:

- **Dev Environment** (`mlflow_artifacts/dev/`): For actual model development (default, port 5001)
- **Quickstart Environment** (`mlflow_artifacts/quickstart/`): For learning/tutorials (port 5000)

```bash
# View dev experiments (default)
make mlflow-ui

# Or manually specify environment
mlflow ui --backend-store-uri sqlite:///mlflow_artifacts/dev/mlflow.db --port 5001
```

## DVC Data Versioning

Data is tracked with DVC and excluded from git:

```bash
# Pull data from remote storage (if configured)
dvc pull

# Track changes to data
dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc
```

See `.dvc/config` for remote storage configuration.

## Project Organization

```
├── LICENSE              <- Open-source license if one is chosen
├── Makefile             <- Makefile with convenience commands like `make preprocess` or `make train`
├── README.md            <- The top-level README for developers using this project
│
├── config/              <- Configuration files
│   └── models.yaml      <- Model training configuration (models, hyperparameters, MLflow settings)
│
├── data/                <- Data directory (managed by DVC, excluded from git)
│   ├── raw/             <- The original, immutable data dump
│   └── processed/       <- The final, canonical data sets for modeling (train/val/test splits)
│
├── mlflow/              <- MLflow tracking databases and artifacts
│   ├── dev/             <- Development environment (default)
│   └── quickstart/      <- Learning/tutorial environment
│
├── models/              <- Trained and serialized models (.pkl files)
│
├── notebooks/           <- Jupyter notebooks for experimentation and EDA
│   ├── 02_clean_preprocess.ipynb
│   ├── news_online_preprocess.ipynb
│   ├── online_news_ml_models.ipynb
│   ├── mlflow-quickstart.ipynb
│   └── ...
│
├── docs/                <- Documentation + data profiling reports
│   ├── docs/            <- MkDocs documentation source
│   │   ├── index.md
│   │   └── getting-started.md
│   ├── mkdocs.yml       <- MkDocs configuration
│   └── *.html           <- Generated data profiling reports
│
├── reports/             <- Generated analysis outputs
│   └── figures/         <- Generated graphics and visualizations
│
├── tests/               <- Unit tests (pytest)
│   └── test_data.py
│
├── .dvc/                <- DVC configuration for data versioning
│
├── pyproject.toml       <- Project configuration with package metadata and tool settings
├── requirements.txt     <- Python dependencies for reproducing the environment
├── setup.cfg            <- Configuration for flake8, isort, black
│
└── mlops_online_news_popularity/   <- Main source package
    │
    ├── __init__.py              <- Makes mlops_online_news_popularity a Python module
    ├── config.py                <- Centralized configuration (paths, logger, MLflow URIs)
    │
    ├── preprocessing/           <- Data preprocessing modules (model-agnostic)
    │   ├── __init__.py
    │   ├── data_processor.py    <- DataProcessor: main preprocessing pipeline
    │   ├── data_cleaning.py     <- DataCleaner: cleaning utilities
    │   ├── data_exploration.py  <- DataExplorer: EDA and profiling
    │   ├── data_io.py           <- DataLoader: CSV operations
    │   ├── data_comparison.py   <- DataComparator: dataset comparison
    │   └── utils.py             <- Utility functions
    │
    ├── modeling/                <- Model training and evaluation (model-specific)
    │   ├── __init__.py
    │   ├── train.py             <- ModelTrainer: training with sklearn Pipeline
    │   ├── compare.py           <- Experimento: MLflow-based model comparison
    │   └── predict.py           <- Model inference
    │
    └── cli/                     <- Command-line interfaces
        ├── __init__.py
        ├── preprocess_cli.py    <- CLI for preprocessing pipeline
        └── train_cli.py         <- CLI for model training with MLflow tracking
```

## Documentation

### Online Documentation

Once deployed to GitHub Pages, view the full documentation at:
- **https://\<username\>.github.io/mlops-project/** (Update with your GitHub username)

### Local Documentation

```bash
# Serve documentation locally (http://localhost:8000)
make docs-serve

# Build documentation
make docs

# Deploy to GitHub Pages
make docs-deploy
```

### Documentation Contents

- **README.md** (this file): Project overview and quick start
- **docs/**: MkDocs documentation source
  - `index.md`: Project overview and features
  - `getting-started.md`: Installation and setup guide
  - `data-profiling.md`: Links to data profiling reports
  - `assets/html/`: Generated profiling reports (pandas-profiling)
  - `assets/images/`: Correlation matrices and visualizations

### Automated Deployment

Documentation is automatically deployed to GitHub Pages when you push to the `main` branch via GitHub Actions.

## Development

### Code Quality

```bash
# Check code quality
make lint

# Auto-format code (black + isort)
make format

# Run tests
make test
```

### Code Style
- **Black**: Line length 99
- **isort**: Black-compatible import sorting
- **flake8**: Linting with specific ignores (E731, E266, E501, C901, W503)
- **Type hints**: Encouraged for better code documentation

## Technologies

- **Python**: 3.10
- **ML Framework**: scikit-learn
- **Experiment Tracking**: MLflow
- **Data Versioning**: DVC
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Data Profiling**: pandas-profiling
- **CLI**: typer
- **Logging**: loguru
- **Testing**: pytest
- **Code Quality**: black, isort, flake8

--------

