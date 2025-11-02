# Preprocessing Utilities

Helper classes and functions for preprocessing.

## DataExplorer

EDA and profiling reports.

```python
from mlops_online_news_popularity.preprocessing import DataExplorer

# Basic EDA
DataExplorer.explore_data(df)

# Correlation heatmap
DataExplorer.plot_correlation_matrix(df, title="Correlation Matrix")

# Profiling report
DataExplorer.generate_profiling_report(df, title="Data Report", output_dir="docs")
```

## DataLoader

CSV I/O operations.

```python
from mlops_online_news_popularity.preprocessing import DataLoader

loader = DataLoader()
df = loader.load_csv("data/raw/dataset.csv")
```

## DataComparator

Compare datasets before/after preprocessing.

```python
from mlops_online_news_popularity.preprocessing import DataComparator

comparator = DataComparator(original_df, cleaned_df)
report = comparator.compare_stats().export_report("report.csv")
```

## Utils

```python
from mlops_online_news_popularity.preprocessing.utils import classify_numeric_columns

binary_cols, non_binary_cols = classify_numeric_columns(df)
```
