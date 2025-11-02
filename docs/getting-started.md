Getting Started
===============

## Prerequisites

- Python 3.10
- pip package manager
- Git

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mlops-project
   ```

2. **Create virtual environment (recommended)**
   ```bash
   make create_environment
   # Or manually:
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the package in development mode**
   ```bash
   pip install -e .
   ```

   This step is **required** for imports to work correctly.

4. **Install dependencies**
   ```bash
   make requirements
   # Or manually:
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Preprocess Data

Run the preprocessing pipeline to create train/val/test splits:

```bash
make preprocess
```

This will:
- Load raw data from `data/raw/online_news_modified.csv`
- Clean and transform the data
- Create train/validation/test splits (70/15/15)
- Save processed datasets to `data/processed/`
- Generate data profiling reports in `docs/`

### 2. Train Models

Train multiple models and compare them with MLflow:

```bash
make train
```

This will:
- Train all models defined in `config/models.yaml`
- Track experiments with MLflow
- Save the best model to `models/`

### 3. View Results

Start the MLflow UI to view experiment results:

```bash
make mlflow-ui
```

Then open http://localhost:5001 in your browser.

## Full Pipeline

Run preprocessing + training in one command:

```bash
make pipeline
```

## Available Make Commands

- `make requirements` - Install dependencies
- `make preprocess` - Run preprocessing pipeline
- `make train` - Train all models from config
- `make train-single` - Train a single model (default: Ridge)
- `make pipeline` - Run full pipeline (preprocess + train)
- `make mlflow-ui` - Start MLflow UI
- `make lint` - Run code quality checks
- `make format` - Auto-format code with black and isort
- `make test` - Run pytest tests
- `make clean` - Remove compiled Python files
- `make help` - Show all available commands
