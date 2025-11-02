# Preprocessing API Reference

Complete API reference for preprocessing modules.

## DataProcessor

**Class**: `mlops_online_news_popularity.preprocessing.DataProcessor`

### Constructor

```python
DataProcessor(
    filepath: str,
    target_col: str = "shares",
    cols_to_drop: Optional[List[str]] = None,
    lda_cols: Optional[List[str]] = None,
    correlation_threshold: float = 0.9
)
```

### Methods

#### `process() -> DataProcessor`
Execute complete preprocessing pipeline.

#### `load_and_clean() -> pd.DataFrame`
Load and clean raw data.

## DataCleaner

**Class**: `mlops_online_news_popularity.preprocessing.DataCleaner`

### Methods

All methods return `self` for chaining.

- `clean_primary_key(key: str) -> DataCleaner`
- `force_numeric(exclude: List[str] = None) -> DataCleaner`
- `apply_business_rules() -> DataCleaner`
- `normalize_lda(lda_cols: List[str]) -> DataCleaner`
- `get_df() -> pd.DataFrame`

## DataExplorer

**Class**: `mlops_online_news_popularity.preprocessing.DataExplorer`

Static methods for EDA.

- `explore_data(data: pd.DataFrame) -> None`
- `plot_correlation_matrix(data: pd.DataFrame, title: str, save_path: str = None) -> None`
- `generate_profiling_report(data: pd.DataFrame, title: str, output_dir: str, filename: str) -> None`

## Utils

**Function**: `classify_numeric_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]`

Returns: (binary_cols, non_binary_cols)
