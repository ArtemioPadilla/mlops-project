# Contributing Guide

Thank you for your interest in contributing to the MLOps Online News Popularity project!

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/mlops-project.git
cd mlops-project
```

### 2. Create Development Environment

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in editable mode
pip install -e .

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install black isort flake8 pytest mkdocs mkdocs-material mkdocs-mermaid2-plugin
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

---

## Code Quality Standards

This project follows strict code quality standards.

### Style Guide

- **Line length**: 99 characters (enforced by Black)
- **Import sorting**: isort with Black profile
- **Linting**: flake8 with specific ignores (E731, E266, E501, C901, W503)
- **Type hints**: Encouraged for function signatures

### Auto-formatting

Before committing, format your code:

```bash
make format  # Runs isort + black
make lint    # Checks flake8
```

**What `make format` does**:
```bash
isort mlops_online_news_popularity
black mlops_online_news_popularity
```

### Manual Formatting

```bash
# Format a specific file
black mlops_online_news_popularity/preprocessing/data_processor.py
isort mlops_online_news_popularity/preprocessing/data_processor.py

# Check without modifying
black --check mlops_online_news_popularity
flake8 mlops_online_news_popularity
```

---

## Testing

### Running Tests

```bash
# Run all tests
make test

# Or directly
pytest tests/

# Run specific test file
pytest tests/test_data.py

# Run with coverage
pytest --cov=mlops_online_news_popularity tests/
```

### Writing Tests

Place tests in `tests/` directory:

```python
# tests/test_preprocessing.py
import pytest
from mlops_online_news_popularity.preprocessing import DataProcessor

def test_data_processor_initialization():
    """Test DataProcessor initializes correctly."""
    processor = DataProcessor(
        filepath="data/raw/test_data.csv",
        target_col="shares"
    )
    assert processor.target_col == "shares"
    assert processor.correlation_threshold == 0.9

def test_binary_classification():
    """Test binary column classification."""
    from mlops_online_news_popularity.preprocessing.utils import classify_numeric_columns
    import pandas as pd

    df = pd.DataFrame({
        'binary_col': [0, 1, 0, 1],
        'continuous_col': [1.5, 2.3, 4.1, 5.9]
    })

    binary, non_binary = classify_numeric_columns(df)

    assert 'binary_col' in binary
    assert 'continuous_col' in non_binary
```

---

## Adding New Features

### Adding a New Model

1. **Update `config/models.yaml`**:
```yaml
models_to_try:
  MyNewModel:
    class_path: "sklearn.ensemble.GradientBoostingRegressor"
    # Optional: add model parameters
    n_estimators: 100
    learning_rate: 0.1
```

2. **Test the model**:
```bash
make train
```

3. **Verify in MLflow UI**:
```bash
make mlflow-ui
```

### Adding a New Preprocessing Step

1. **Add method to appropriate class**:
```python
# mlops_online_news_popularity/preprocessing/data_cleaning.py

class DataCleaner:
    # ... existing methods ...

    def remove_outliers(self, columns: List[str], n_std: float = 3.0):
        """Remove outliers beyond n standard deviations."""
        for col in columns:
            mean = self.df[col].mean()
            std = self.df[col].std()
            self.df = self.df[
                (self.df[col] >= mean - n_std * std) &
                (self.df[col] <= mean + n_std * std)
            ]
        return self
```

2. **Update `DataProcessor` to use it**:
```python
# In load_and_clean() method
cleaned_df = (cleaner
    .clean_primary_key(key="url")
    .force_numeric(exclude=["url"])
    .apply_business_rules()
    .remove_outliers(['timedelta'])  # NEW
    .normalize_lda(self.lda_cols)
    .get_df())
```

3. **Add tests**:
```python
# tests/test_data_cleaning.py
def test_remove_outliers():
    """Test outlier removal."""
    df = pd.DataFrame({'col': [1, 2, 3, 100]})  # 100 is outlier
    cleaner = DataCleaner(df)
    result = cleaner.remove_outliers(['col'], n_std=2).get_df()
    assert 100 not in result['col'].values
```

---

## Documentation

### Updating Documentation

Documentation is built with MkDocs Material and lives in `docs/`.

```bash
# Serve documentation locally
make docs-serve
# Open http://localhost:8000

# Build documentation
make docs

# Deploy to GitHub Pages
make docs-deploy
```

### Adding a New Documentation Page

1. **Create Markdown file**:
```bash
# Create new page
touch docs/new-section/my-page.md
```

2. **Update `mkdocs.yml` navigation**:
```yaml
nav:
  # ... existing sections ...
  - New Section:
      - My Page: new-section/my-page.md
```

3. **Add content with examples**:
```markdown
# My New Page

Description here.

## Code Example

\`\`\`python
from mlops_online_news_popularity import something
something.do_thing()
\`\`\`

## Mermaid Diagram

\`\`\`mermaid
graph LR
    A[Start] --> B[Process]
    B --> C[End]
\`\`\`
```

---

## Commit Guidelines

### Commit Message Format

Follow conventional commits:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples**:
```bash
git commit -m "feat(preprocessing): add outlier removal method"
git commit -m "fix(modeling): correct RMSE calculation for log-transformed targets"
git commit -m "docs(api): add ModelTrainer API reference"
```

### Before Committing

```bash
# 1. Format code
make format

# 2. Run linter
make lint

# 3. Run tests
make test

# 4. Add changes
git add .

# 5. Commit
git commit -m "feat: your feature description"
```

---

## Pull Request Process

### 1. Push Your Branch

```bash
git push origin feature/your-feature-name
```

### 2. Create Pull Request

On GitHub:
1. Click "New Pull Request"
2. Select your branch
3. Fill in the template:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests for new features
- [ ] Documentation updated

## Checklist
- [ ] Code follows project style (ran `make format`)
- [ ] Linting passes (ran `make lint`)
- [ ] All tests pass (ran `make test`)
- [ ] Documentation updated (if needed)
```

### 3. Code Review

- Address reviewer feedback
- Update your branch as needed:
```bash
git add .
git commit -m "fix: address review comments"
git push origin feature/your-feature-name
```

### 4. Merge

Once approved, your PR will be merged!

---

## Project Structure Guidelines

### Where to Add Code

```
mlops_online_news_popularity/
├── preprocessing/          # Data preprocessing (model-agnostic)
│   ├── data_processor.py   # Main orchestrator
│   ├── data_cleaning.py    # Cleaning utilities
│   ├── data_exploration.py # EDA and profiling
│   └── utils.py            # Helper functions
├── modeling/               # Model training (model-specific)
│   ├── train.py            # ModelTrainer class
│   ├── compare.py          # Multi-model comparison
│   └── predict.py          # Inference
└── cli/                    # Command-line interfaces
    ├── preprocess_cli.py   # Preprocessing CLI
    └── train_cli.py        # Training CLI
```

**Guidelines**:
- Model-agnostic preprocessing → `preprocessing/`
- Model-specific operations → `modeling/`
- CLI commands → `cli/`
- Utility functions → `**/utils.py`
- Configuration → `config.py`

---

## Code Review Checklist

As a reviewer, check for:

- [ ] Code follows Black/isort formatting
- [ ] No flake8 violations
- [ ] Tests added for new features
- [ ] Documentation updated
- [ ] No data leakage (train/test separation)
- [ ] Type hints used for public functions
- [ ] Logging used instead of print statements
- [ ] Error handling for edge cases
- [ ] No hardcoded paths (use config.py)

---

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue
- **Feature Requests**: Open a GitHub Issue with `enhancement` label

Thank you for contributing! 🎉
