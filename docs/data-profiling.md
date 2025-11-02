# Data Profiling Reports

This page provides access to comprehensive data profiling reports generated during the preprocessing pipeline. These reports are automatically created using pandas-profiling and provide detailed insights into the dataset at various stages.

## Overview

The preprocessing pipeline generates profiling reports at key stages:

1. **Raw Data Report**: Analysis of the original dataset before any transformations
2. **Cleaned Data Report**: Analysis after data cleaning and business rule applications
3. **Train Set Report**: Statistics and distributions of the training set
4. **Test Set Report**: Statistics and distributions of the test set

## Profiling Reports

### [Raw Data Report](assets/html/01_raw_data_report.html)

Complete analysis of the original dataset including:

- Dataset shape and size
- Missing values analysis
- Variable types and distributions
- Correlations between features
- Warnings and data quality issues

[**View Raw Data Report →**](assets/html/01_raw_data_report.html)

---

### [Cleaned Data Report](assets/html/02_cleaned_data_report.html)

Analysis after applying data cleaning transformations:

- URL validation and cleaning
- Numeric type conversion
- Business rules application (timedelta clipping, LDA normalization)
- Comparison with raw data

[**View Cleaned Data Report →**](assets/html/02_cleaned_data_report.html)

---

### [Train Set Report](assets/html/03_train_set_report.html)

Detailed statistics for the training set (70% of data):

- Feature distributions
- Target variable analysis (shares)
- Correlations within training data
- Data quality checks

[**View Train Set Report →**](assets/html/03_train_set_report.html)

---

### [Test Set Report](assets/html/04_test_set_report.html)

Detailed statistics for the test set (15% of data):

- Feature distributions
- Comparison with training set
- Data leakage checks
- Distribution consistency

[**View Test Set Report →**](assets/html/04_test_set_report.html)

---

## Correlation Analysis

### Correlation Matrix Before Feature Selection

This heatmap shows correlations between all features before removing highly correlated variables.

![Correlation Matrix Before](assets/images/05_corr_matrix_before.png)

*Figure 1: Correlation matrix of all features in the original dataset. Dark colors indicate strong correlations.*

---

### Correlation Matrix After Feature Selection

This heatmap shows the remaining features after removing those with correlation > 0.9 (configurable threshold).

![Correlation Matrix After](assets/images/06_corr_matrix_after.png)

*Figure 2: Correlation matrix after removing highly correlated features. This helps prevent multicollinearity issues in modeling.*

---

## Generating These Reports

These reports can be generated using the `DataExplorer` class from the preprocessing module:

```python
from mlops_online_news_popularity.preprocessing import DataExplorer
from mlops_online_news_popularity.config import PROFILING_REPORTS_DIR, PROFILING_IMAGES_DIR
import pandas as pd

# Load your data
df = pd.read_csv("data/raw/online_news_modified.csv")

# Generate profiling report
DataExplorer.generate_profiling_report(
    data=df,
    title="My Data Report",
    output_dir=str(PROFILING_REPORTS_DIR),
    filename="my_report.html"
)

# Generate correlation heatmap
DataExplorer.plot_correlation_matrix(
    data=df,
    title="Correlation Matrix",
    save_path=str(PROFILING_IMAGES_DIR / "correlation_matrix.png")
)
```

The reports are saved to `docs/assets/html/` and images to `docs/assets/images/`. Both can be viewed in any web browser.

!!! note "Path Constants"
    Use `PROFILING_REPORTS_DIR` and `PROFILING_IMAGES_DIR` from `mlops_online_news_popularity.config` to ensure reports are saved to the correct location for MkDocs documentation.

## Understanding the Reports

### Key Sections in Profiling Reports

- **Overview**: High-level statistics (number of variables, observations, missing cells)
- **Variables**: Detailed analysis of each feature (type, distribution, missing values, unique values)
- **Interactions**: Pairwise interactions between variables
- **Correlations**: Correlation matrices using different methods (Pearson, Spearman, Kendall)
- **Missing Values**: Visualization and analysis of missing data patterns
- **Alerts**: Warnings about data quality issues (high cardinality, skewness, zeros, etc.)

### Tips for Interpretation

1. **Check Alerts**: Start with the alerts section to identify immediate data quality issues
2. **Review Distributions**: Look for skewed distributions that might need transformation
3. **Analyze Correlations**: Identify highly correlated features for potential removal
4. **Missing Values**: Decide on imputation strategies based on missing patterns
5. **Compare Reports**: Use before/after reports to validate preprocessing steps

---

## Next Steps

After reviewing the profiling reports:

1. [Train models](getting-started.md#2-train-models) using the preprocessed data
2. [View MLflow experiments](getting-started.md#3-view-results) to compare model performance
3. Iterate on preprocessing pipeline based on insights from the reports
