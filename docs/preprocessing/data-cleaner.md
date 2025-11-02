# DataCleaner Class

Builder pattern implementation for data cleaning with method chaining.

## Purpose

`DataCleaner` provides a fluent interface for applying data cleaning operations.

## Usage

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

## Methods

### `clean_primary_key(key: str)`
Validate and clean primary key column.

### `force_numeric(exclude: List[str])`
Convert string numbers to numeric type.

### `apply_business_rules()`
Apply domain-specific rules (timedelta clipping, etc.).

### `normalize_lda(lda_cols: List[str])`
Normalize LDA topic columns to sum to 1.

### `get_df()`
Return the cleaned DataFrame.

See [API Reference](../api-reference/preprocessing.md) for details.
