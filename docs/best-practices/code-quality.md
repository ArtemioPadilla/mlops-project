# Code Quality Standards

Standards and tools for maintaining code quality.

## Formatting

### Black

Line length: 99 characters

```bash
black mlops_online_news_popularity
```

### isort

Import sorting compatible with Black:

```bash
isort mlops_online_news_popularity
```

### Combined

```bash
make format  # Runs both isort and black
```

## Linting

### flake8

Configuration in `setup.cfg`:

```ini
[flake8]
max-line-length = 99
ignore = E731,E266,E501,C901,W503
exclude = .git,notebooks,references,models,data
```

```bash
make lint
```

## Type Hints

Encouraged for public functions:

```python
def process_data(df: pd.DataFrame, target_col: str = "shares") -> Tuple[pd.DataFrame, pd.Series]:
    """Process data and return features and target."""
    ...
```

## Logging

Use loguru instead of print:

```python
from loguru import logger

logger.info("Processing started")
logger.debug(f"Shape: {df.shape}")
logger.warning("Missing values found")
logger.error("Processing failed")
```

## Documentation

- Docstrings for all public functions/classes
- Type hints where applicable
- Inline comments for complex logic
