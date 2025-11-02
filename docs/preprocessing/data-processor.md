# DataProcessor Class

The `DataProcessor` class orchestrates the complete model-agnostic preprocessing pipeline.

## Purpose

`DataProcessor` handles all preprocessing steps that are independent of the specific ML model being used.

## Basic Usage

```python
from mlops_online_news_popularity.preprocessing import DataProcessor

processor = DataProcessor(
    filepath='data/raw/online_news_modified.csv',
    target_col='shares',
    correlation_threshold=0.9
)

# Run complete preprocessing
processor.process()

# Access clean splits
X_train, y_train = processor.X_train, processor.y_train
X_val, y_val = processor.X_val, processor.y_val
X_test, y_test = processor.X_test, processor.y_test

# Access metadata
binary_cols = processor.cols_bin
non_binary_cols = processor.cols_no_bin
```

## Class Reference

Located in: `mlops_online_news_popularity/preprocessing/data_processor.py`

### Initialization Parameters

- **filepath** (str): Path to raw CSV file
- **target_col** (str): Name of target column (default: "shares")
- **cols_to_drop** (List[str]): Non-predictive columns (default: ['url', 'timedelta'])
- **lda_cols** (List[str]): LDA topic columns to normalize
- **correlation_threshold** (float): Threshold for removing correlated features (default: 0.9)

### Attributes After `process()`

**Data Splits**:
- `X_train`, `X_val`, `X_test`: Feature DataFrames
- `y_train`, `y_val`, `y_test`: Target Series

**Metadata**:
- `cols_bin`: List of binary columns
- `cols_no_bin`: List of non-binary columns
- `cols_dropped_correlation`: Features removed due to high correlation
- `numeric_features`: All numeric feature names

## Methods

### `process()`

Execute the complete preprocessing pipeline.

**Returns**: `self` (for method chaining)

**Steps**:
1. Load and clean data (`load_and_clean()`)
2. Engineer features (`engineer_features()`)
3. Split into train/val/test (`split_data()`)
4. Handle high correlation (`_handle_high_correlation()`)

### `load_and_clean()`

Load raw CSV and apply data cleaning.

**Returns**: Clean DataFrame

**What it does**:
- Creates `DataCleaner` instance
- Applies cleaning methods (URL cleaning, numeric conversion, business rules, LDA normalization)
- Returns cleaned DataFrame

See [DataProcessor API](../api-reference/preprocessing.md) for complete reference.
