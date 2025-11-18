"""
Data processor module for model-agnostic preprocessing and data splitting.

This module provides the DataProcessor class which handles all preprocessing steps
that are independent of the specific model being used, including data cleaning,
feature engineering, and train/val/test splitting.
"""

from typing import List, Optional, Tuple

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data_cleaning import DataCleaner
from .utils import classify_numeric_columns


class DataProcessor:
    """
    Handles all model-agnostic preprocessing and data splitting.

    This class orchestrates the complete preprocessing pipeline from raw data
    to clean train/val/test splits ready for modeling.

    Workflow:
    ---------
    1. Load and clean raw data (using DataCleaner)
    2. Engineer features (drop non-predictive columns, classify types)
    3. Split into train/val/test sets
    4. Handle high correlation (on train set only, to prevent data leakage)

    Output:
    -------
    Clean train/val/test splits with metadata about column types for ModelTrainer.

    Example:
    --------
    >>> processor = DataProcessor(
    ...     filepath='data/raw/online_news_modified.csv',
    ...     target_col='shares'
    ... )
    >>> processor.process()
    >>> # Access splits
    >>> X_train, y_train = processor.X_train, processor.y_train
    >>> # Access column classifications
    >>> binary_cols = processor.cols_bin
    """

    def __init__(
        self,
        filepath: str,
        target_col: str = "shares",
        cols_to_drop: Optional[List[str]] = None,
        lda_cols: Optional[List[str]] = None,
        correlation_threshold: float = 0.9,
    ):
        """
        Initialize DataProcessor.

        Parameters
        ----------
        filepath : str
            Path to the raw CSV file
        target_col : str, optional
            Name of the target column (default: 'shares')
        cols_to_drop : List[str], optional
            Non-predictive columns to drop (default: ['url', 'timedelta'])
        lda_cols : List[str], optional
            LDA topic columns to normalize (default: ['LDA_00', ..., 'LDA_04'])
        correlation_threshold : float, optional
            Threshold for removing highly correlated features (default: 0.9)
        """
        self.filepath = filepath
        self.target_col = target_col
        self.cols_to_drop = cols_to_drop or ["url", "timedelta"]
        self.lda_cols = lda_cols or ["LDA_00", "LDA_01", "LDA_02", "LDA_03", "LDA_04"]
        self.correlation_threshold = correlation_threshold

        # Will be populated by process()
        self.X_train: Optional[pd.DataFrame] = None
        self.X_val: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_val: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

        # Column classifications (for ModelTrainer)
        self.cols_bin: List[str] = []
        self.cols_no_bin: List[str] = []
        self.numeric_features: List[str] = []

        # Dropped columns tracking
        self.cols_dropped_correlation: List[str] = []

    def load_and_clean(self) -> pd.DataFrame:
        """
        Load raw data and apply cleaning transformations.

        Uses DataCleaner to:
        - Clean primary key (URL)
        - Force numeric conversion
        - Apply business rules (timedelta clipping, proportion constraints)
        - Normalize LDA topics

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame
        """
        print("\n" + "=" * 70)
        print("PHASE 1: LOADING AND CLEANING DATA")
        print("=" * 70)

        # Load raw data
        print(f"Loading data from: {self.filepath}")
        df = pd.read_csv(self.filepath)
        print(f"Loaded shape: {df.shape}")

        # Apply cleaning transformations
        cleaner = DataCleaner(df)
        cleaned = (
            cleaner.clean_primary_key(key="url")
            .force_numeric(exclude=["url"])
            .apply_business_rules()
            .normalize_lda(lda_cols=self.lda_cols)
            .get_df()
        )

        # Remove rows with missing target values
        if self.target_col in cleaned.columns:
            rows_before = len(cleaned)
            cleaned = cleaned[cleaned[self.target_col].notna()]
            rows_removed = rows_before - len(cleaned)
            if rows_removed > 0:
                print(f"Removed {rows_removed} rows with missing target '{self.target_col}'")

        print(f"Cleaned shape: {cleaned.shape}")
        print("=" * 70)

        return cleaned

    def engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Perform model-agnostic feature engineering.

        Steps:
        ------
        1. Separate features (X) and target (y)
        2. Drop non-predictive columns (url, timedelta)
        3. Identify numeric features
        4. Classify binary vs non-binary columns

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned DataFrame

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            Features (X) and target (y)
        """
        print("\n" + "=" * 70)
        print("PHASE 2: FEATURE ENGINEERING")
        print("=" * 70)

        # Separate X and y
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in DataFrame")

        X = df.drop(self.target_col, axis=1)
        y = df[self.target_col]

        # Remove rows where target is NaN
        nan_mask = y.isna()
        nan_count = nan_mask.sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} rows with NaN in target '{self.target_col}'")
            valid_mask = ~nan_mask
            X = X[valid_mask]
            y = y[valid_mask]
            logger.info(f"Removed {nan_count} rows with NaN in target. New shape: {y.shape}")

        print(f"Target: {self.target_col}")
        print(f"Target shape: {y.shape}")
        print(
            f"Target stats: min={y.min():.2f}, median={y.median():.2f}, "
            f"max={y.max():.2f}, skew={y.skew():.2f}"
        )

        # Drop non-predictive columns
        cols_to_drop_present = [col for col in self.cols_to_drop if col in X.columns]
        if cols_to_drop_present:
            X = X.drop(columns=cols_to_drop_present)
            print(f"\nDropped non-predictive columns: {cols_to_drop_present}")

        # Identify numeric features
        self.numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        print(f"Numeric features: {len(self.numeric_features)}")

        # Classify binary vs non-binary columns
        self.cols_bin, self.cols_no_bin = classify_numeric_columns(X[self.numeric_features])

        print("\nFeature classification:")
        print(f"  Binary columns: {len(self.cols_bin)}")
        print(f"  Non-binary columns: {len(self.cols_no_bin)}")
        print(f"Total features after engineering: {X.shape[1]}")
        print("=" * 70)

        return X, y

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_size: float = 0.70,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train/val/test sets.

        Strategy: Sequential split
        1. Split into train vs (val + test)
        2. Split (val + test) into val vs test

        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        train_size : float, optional
            Proportion for training set (default: 0.70)
        val_size : float, optional
            Proportion for validation set (default: 0.15)
        test_size : float, optional
            Proportion for test set (default: 0.15)
        random_state : int, optional
            Random seed for reproducibility (default: 42)

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        print("\n" + "=" * 70)
        print("PHASE 3: TRAIN/VAL/TEST SPLIT")
        print("=" * 70)

        # Validate proportions
        total = train_size + val_size + test_size
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split proportions must sum to 1.0, got {total:.3f}")

        # First split: train vs (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, train_size=train_size, random_state=random_state, shuffle=True
        )

        # Second split: val vs test
        val_ratio = val_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, train_size=val_ratio, random_state=random_state, shuffle=True
        )

        print(f"Split proportions: {train_size:.0%} / {val_size:.0%} / {test_size:.0%}")
        print(f"Train set: {X_train.shape}")
        print(f"Val set:   {X_val.shape}")
        print(f"Test set:  {X_test.shape}")
        print("=" * 70)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _handle_high_correlation(self, X_train: pd.DataFrame, threshold: float) -> List[str]:
        """
        Identify and remove highly correlated features.

        Strategy:
        ---------
        For each pair of features with correlation > threshold:
        - Keep the one with LOWER average correlation (more generalizable)
        - Drop the one with HIGHER average correlation

        Note: This is done on train set only to prevent data leakage.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        threshold : float
            Correlation threshold (e.g., 0.9)

        Returns
        -------
        List[str]
            Column names to drop
        """
        print("\n" + "=" * 70)
        print("PHASE 4: HANDLING HIGH CORRELATION")
        print("=" * 70)
        print(f"Correlation threshold: {threshold}")

        corr_matrix = X_train.corr(numeric_only=True).abs()
        to_drop = set()

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col_i, col_j = corr_matrix.columns[i], corr_matrix.columns[j]

                if col_i in to_drop or col_j in to_drop:
                    continue

                if corr_matrix.iloc[i, j] > threshold:
                    # Drop the one with HIGHER average correlation
                    avg_corr_i = corr_matrix[col_i].mean()
                    avg_corr_j = corr_matrix[col_j].mean()

                    col_to_drop = col_i if avg_corr_i > avg_corr_j else col_j
                    to_drop.add(col_to_drop)

                    print(
                        f"  High correlation ({corr_matrix.iloc[i, j]:.3f}): "
                        f"{col_i} <-> {col_j}"
                    )
                    print(
                        f"    → Dropping '{col_to_drop}' "
                        f"(avg_corr={avg_corr_i if col_to_drop == col_i else avg_corr_j:.3f})"
                    )

        cols_to_drop = list(to_drop)
        print(f"\nTotal columns to drop: {len(cols_to_drop)}")
        print("=" * 70)

        return cols_to_drop

    def process(self) -> "DataProcessor":
        """
        Execute complete preprocessing pipeline.

        Pipeline:
        ---------
        1. Load and clean raw data
        2. Engineer features
        3. Split into train/val/test
        4. Handle high correlation (on train set only)

        Returns
        -------
        DataProcessor
            Self (for method chaining)
        """
        # Phase 1: Load and clean
        df_cleaned = self.load_and_clean()

        # Phase 2: Feature engineering
        X, y = self.engineer_features(df_cleaned)

        # Phase 3: Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)

        # Phase 4: Handle high correlation (on train set only!)
        self.cols_dropped_correlation = self._handle_high_correlation(
            X_train, self.correlation_threshold
        )

        # Apply correlation-based drops to all splits
        if self.cols_dropped_correlation:
            X_train = X_train.drop(columns=self.cols_dropped_correlation, errors="ignore")
            X_val = X_val.drop(columns=self.cols_dropped_correlation, errors="ignore")
            X_test = X_test.drop(columns=self.cols_dropped_correlation, errors="ignore")

            # Update column classifications
            self.cols_bin = [c for c in self.cols_bin if c not in self.cols_dropped_correlation]
            self.cols_no_bin = [
                c for c in self.cols_no_bin if c not in self.cols_dropped_correlation
            ]

        # Store final splits
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        # Summary
        print("\n" + "=" * 70)
        print("PREPROCESSING COMPLETE")
        print("=" * 70)
        print(f"Final feature count: {X_train.shape[1]}")
        print(f"  Binary: {len(self.cols_bin)}")
        print(f"  Non-binary: {len(self.cols_no_bin)}")
        print("\nData splits:")
        print(f"  Train: {X_train.shape}")
        print(f"  Val:   {X_val.shape}")
        print(f"  Test:  {X_test.shape}")
        print("=" * 70 + "\n")

        return self
