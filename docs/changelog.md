# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation with MkDocs Material
- Mermaid diagrams for architecture and data flow
- GitHub Pages deployment workflow
- 20+ documentation pages covering all aspects
- Troubleshooting guide
- Contributing guidelines

### Changed
- Moved mkdocs.yml to project root (standard practice)
- Reorganized docs/ directory with assets subdirectories
- Updated .gitignore for MkDocs site/ directory

---

## [1.0.0] - 2024-11-02

### Added
- Complete MLOps pipeline for Online News Popularity prediction
- Model-agnostic preprocessing with `DataProcessor`
- Model-specific training with `ModelTrainer`
- MLflow experiment tracking with parent-child run hierarchy
- Multi-model comparison with `Experimento` class
- CLI interfaces for preprocessing and training
- DVC configuration for data versioning
- Automated data profiling reports with pandas-profiling
- Makefile commands for common workflows
- Two MLflow environments (dev and quickstart)

### Features

#### Preprocessing Module
- `DataProcessor`: Complete preprocessing pipeline
  - Data cleaning with method chaining (`DataCleaner`)
  - Binary vs non-binary column classification
  - 70/15/15 train/val/test splitting
  - Correlation-based feature selection (train-only)
- `DataExplorer`: EDA and profiling report generation
- `DataLoader`: CSV I/O operations
- `DataComparator`: Dataset comparison utilities

#### Modeling Module
- `ModelTrainer`: sklearn Pipeline-based training
  - Separate preprocessing for binary and non-binary features
  - Optional target transformation (log1p)
  - Comprehensive evaluation with diagnostics
  - Cross-validation support
- `Experimento`: Multi-model experiment tracking
  - YAML-based model configuration
  - MLflow parent-child run hierarchy
  - Automatic best model selection
  - Model artifact saving

#### CLI Commands
- `preprocess_cli`: Run preprocessing pipeline
  - Configurable input/output paths
  - Correlation threshold parameter
  - Saves splits and metadata to disk
- `train_cli`: Model training with MLflow
  - `train-compare`: Multi-model comparison
  - `train-single`: Quick single-model training

#### Configuration
- Centralized path management in `config.py`
- MLflow tracking URI configuration
- Environment variable overrides via `.env`
- Model configuration via YAML

#### Code Quality
- Black formatting (line length 99)
- isort import sorting
- flake8 linting
- pytest testing framework
- Type hints for major functions

---

## [0.2.0] - 2024-10-15

### Added
- Refactored to composition-based architecture
- Separated concerns: model-agnostic vs model-specific
- `DataProcessor` class for unified preprocessing
- `ModelTrainer` class for training with sklearn Pipeline

### Changed
- Migrated from standalone scripts to modular classes
- Consolidated preprocessing into single module
- Improved data leakage prevention

### Deprecated
- Old standalone scripts (`dataset.py`, `features.py`, `plots.py`)

---

## [0.1.0] - 2024-09-20

### Added
- Initial project structure using Cookiecutter Data Science
- Basic data cleaning scripts
- Exploratory data analysis notebooks
- Simple model training scripts
- Basic MLflow integration

---

## Migration Guides

### Migrating from 0.1.0 to 1.0.0

**Old approach**:
```python
# Multiple scattered scripts
from dataset import load_data, clean_data
from features import engineer_features
from plots import plot_distributions

df = load_data()
df = clean_data(df)
X, y = engineer_features(df)
# ...manual splitting and training
```

**New approach**:
```python
# Unified pipeline
from mlops_online_news_popularity.preprocessing import DataProcessor
from mlops_online_news_popularity.modeling.train import ModelTrainer

# Preprocessing
processor = DataProcessor(filepath='data/raw/online_news_modified.csv')
processor.process()

# Training
trainer = ModelTrainer(processor, estimator, "MyModel")
trainer.train_model()
metrics = trainer.evaluate_model()
```

**Benefits**:
- ✅ No data leakage
- ✅ Reproducible pipelines
- ✅ MLflow tracking
- ✅ Easier to swap models

---

## Deprecation Notices

### Deprecated in 1.0.0

The following files are deprecated and will be removed in 2.0.0:

- `mlops_online_news_popularity/dataset.py` → Use `preprocessing.DataProcessor`
- `mlops_online_news_popularity/features.py` → Use `preprocessing.DataProcessor`
- `mlops_online_news_popularity/plots.py` → Use `preprocessing.DataExplorer`

**Migration timeline**:
- 1.0.0: Deprecated (warnings shown)
- 1.5.0: Deprecated (errors shown)
- 2.0.0: Removed

---

## Roadmap

### Planned for 1.1.0
- [ ] Hyperparameter tuning with Optuna
- [ ] Model serving with Flask/FastAPI
- [ ] Docker containerization
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Model performance monitoring
- [ ] A/B testing framework

### Planned for 1.2.0
- [ ] Feature store integration
- [ ] Real-time inference API
- [ ] Model explainability (SHAP values)
- [ ] Automated retraining pipeline
- [ ] Cloud deployment (AWS/GCP/Azure)

### Planned for 2.0.0
- [ ] Complete removal of deprecated modules
- [ ] Refactor to support deep learning models
- [ ] Distributed training support
- [ ] Advanced feature engineering pipeline

---

## Contributors

- **Artemio Padilla** - Initial work and architecture
- **equipo-27** - Team collaboration

---

[Unreleased]: https://github.com/artemiopadilla/mlops-project/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/artemiopadilla/mlops-project/releases/tag/v1.0.0
[0.2.0]: https://github.com/artemiopadilla/mlops-project/releases/tag/v0.2.0
[0.1.0]: https://github.com/artemiopadilla/mlops-project/releases/tag/v0.1.0
