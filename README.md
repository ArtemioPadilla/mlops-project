# MLOps: Online News Popularity Prediction

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A complete MLOps pipeline for predicting the popularity (shares) of online news articles using machine learning. Built with scikit-learn, MLflow, and DVC following best practices for reproducible ML.

## Features

- **Automated Data Pipeline**: End-to-end preprocessing with cleaning, feature engineering, and train/val/test splitting
- **MLflow Experiment Tracking**: Compare multiple models with automated metric logging and artifact versioning
- **FastAPI Model Serving**: Production-ready REST API with online and batch prediction endpoints
- **Docker Containerization**: Portable deployment with multi-stage builds and health checks
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

### MLOps Pipeline

| Command | Description |
|---------|-------------|
| `make requirements` | Install dependencies |
| `make preprocess` | Run preprocessing pipeline |
| `make train` | Train all models from config |
| `make train-single` | Train a single model (default: Ridge) |
| `make pipeline` | Run full pipeline (preprocess + train) |
| `make mlflow-ui` | Start MLflow UI (port 5001) |

### Model Serving (API)

| Command | Description |
|---------|-------------|
| `make serve` | Run FastAPI server locally (development with auto-reload) |
| `make serve-prod` | Run FastAPI server (production with 4 workers) |
| `make test-api` | Test single prediction endpoint |
| `make test-api-batch` | Test batch prediction endpoint (JSON) |
| `make test-api-csv` | Test batch prediction endpoint (CSV) |

### Docker Deployment

| Command | Description |
|---------|-------------|
| `make docker-build` | Build Docker image |
| `make docker-run` | Run Docker container |
| `make docker-up` | Start services with docker-compose |
| `make docker-down` | Stop docker-compose services |
| `make docker-logs` | View docker-compose logs |

### Documentation & Development

| Command | Description |
|---------|-------------|
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

## API Serving with FastAPI

The project includes a production-ready FastAPI service for serving trained models via REST API.

### Quick Start - API Server

```bash
# 1. Run locally (development mode with auto-reload)
make serve

# 2. Access the API
# - API Documentation: http://localhost:8000/docs
# - Health Check: http://localhost:8000/health
# - Model Info: http://localhost:8000/info

# 3. Test the API
make test-api         # Single prediction
make test-api-batch   # Batch prediction (JSON)
make test-api-csv     # Batch prediction (CSV)
```

### API Endpoints

#### `GET /health`
Health check endpoint

```bash
curl http://localhost:8000/health
```

#### `GET /info`
Get model information and metadata

```bash
curl http://localhost:8000/info
```

#### `POST /predict`
Single prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/sample_input.json
```

#### `POST /predict/batch`
Batch prediction (JSON)

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d @examples/sample_batch.json
```

#### `POST /predict/batch/csv`
Batch prediction (CSV upload)

```bash
curl -X POST http://localhost:8000/predict/batch/csv \
  -F "file=@examples/sample_data.csv"
```

### Interactive API Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide:
- Complete API schema with request/response examples
- Interactive testing interface
- Download OpenAPI spec (JSON)

### Configuration

Configure the API server via environment variables (`.env` file):

```bash
# Model Configuration
MODEL_NAME=RandomForestBase
MODEL_LOAD_STRATEGY=local  # Options: local, mlflow
MODEL_PATH=models/randomforestbase_best_20251102_165526.pkl

# MLflow Model Loading (if using mlflow strategy)
# MLFLOW_RUN_ID=your-run-id-here
# MLFLOW_TRACKING_URI=sqlite:///mlflow_artifacts/dev/mlflow.db

# API Server Settings
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

## Docker Deployment

### Build and Run with Docker

```bash
# 1. Build Docker image
make docker-build
# Or: docker build -t ml-service:latest .

# 2. Run container
make docker-run
# Or: docker run -p 8000:8000 -v $(pwd)/models:/app/models ml-service:latest

# 3. Check container logs
docker logs online-news-predictor

# 4. Stop container
docker stop online-news-predictor
```

### Docker Compose (Recommended)

```bash
# Start services in background
make docker-up

# View logs
make docker-logs

# Stop services
make docker-down
```

### Docker Image Details

- **Base Image**: python:3.10-slim
- **Size**: ~300MB (with RandomForest model)
- **Architecture**: Multi-stage build for efficiency
- **Security**: Non-root user, minimal dependencies
- **Health Check**: Automatic health monitoring
- **Volumes**: Models and MLflow artifacts mounted externally

### Model Artifact Path

The trained models are stored in:
- **Local**: `models/randomforestbase_best_20251102_165526.pkl`
- **MLflow Registry**: `models:/<ModelName>/<version>` or `runs:/<run_id>/model_pipeline`

Update `MODEL_PATH` or `MLFLOW_RUN_ID` in `.env` to use a different model.

### Publishing to Docker Registry

```bash
# Tag image for registry
docker tag ml-service:latest your-dockerhub-username/ml-service:latest
docker tag ml-service:latest your-dockerhub-username/ml-service:v1.0.0

# Push to Docker Hub
docker push your-dockerhub-username/ml-service:latest
docker push your-dockerhub-username/ml-service:v1.0.0

# Pull and run from registry
docker pull your-dockerhub-username/ml-service:latest
docker run -p 8000:8000 -v $(pwd)/models:/app/models your-dockerhub-username/ml-service:latest
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
├── mlflow_artifacts/    <- MLflow tracking databases and artifacts
│   ├── dev/             <- Development environment (default)
│   └── quickstart/      <- Learning/tutorial environment
│
├── models/              <- Trained and serialized models (.pkl files)
│
├── notebooks/           <- Jupyter notebooks (numbered in workflow order)
│   ├── 01_news_online_eda.ipynb              <- Exploratory data analysis
│   ├── 02_clean_preprocess.ipynb             <- Initial data cleaning
│   ├── 03_news_online_preprocess.ipynb       <- Advanced preprocessing
│   ├── 04_online_news_ml_models.ipynb        <- Model training and evaluation
│   ├── 05_mlflow_quickstart.ipynb            <- MLflow tutorial
│   ├── 06_testing_news_online_popularity.ipynb <- Testing and validation
│   └── archive/                              <- Deprecated notebooks
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
├── examples/            <- Example scripts and sample data for API testing
│   ├── test_predict_single.py   <- Test single prediction endpoint
│   ├── test_predict_batch.py    <- Test batch prediction endpoint (JSON)
│   ├── test_predict_csv.py      <- Test CSV upload endpoint
│   ├── sample_input.json        <- Sample input for single prediction
│   ├── sample_batch.json        <- Sample batch input
│   └── sample_data.csv          <- Sample CSV for batch prediction
│
├── scripts/             <- Utility scripts
│   ├── docker_build.sh          <- Build Docker image with versioning
│   └── docker_run.sh            <- Run Docker container with volume mounts
│
├── Dockerfile           <- Multi-stage Docker build for API service
├── docker-compose.yml   <- Docker Compose configuration
├── .dockerignore        <- Files to exclude from Docker build
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
    ├── serving/                 <- FastAPI model serving (REST API)
    │   ├── __init__.py
    │   ├── app.py               <- FastAPI application with endpoints
    │   ├── model_handler.py     <- ModelHandler: inference with preprocessing
    │   ├── schemas.py           <- Pydantic request/response schemas
    │   └── config.py            <- Serving configuration (env vars)
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
- **API Framework**: FastAPI + uvicorn
- **Containerization**: Docker + Docker Compose
- **Data Processing**: pandas, numpy
- **Validation**: Pydantic
- **Visualization**: matplotlib, seaborn
- **Data Profiling**: pandas-profiling
- **CLI**: typer
- **Logging**: loguru
- **Testing**: pytest
- **Code Quality**: black, isort, flake8

--------

