# Testing

Testing practices and guidelines.

## Running Tests

```bash
# All tests
make test

# Specific test file
pytest tests/test_data.py

# With coverage
pytest --cov=mlops_online_news_popularity tests/
```

## Writing Tests

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

def test_binary_classification():
    """Test binary column classification."""
    # Test implementation
    pass
```

## Test Naming Convention

- Files: `test_*.py`
- Functions: `test_*`
- Classes: `Test*`

## Fixtures

Use pytest fixtures for reusable test data:

```python
@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [0, 1, 0]
    })

def test_with_fixture(sample_dataframe):
    assert len(sample_dataframe) == 3
```
