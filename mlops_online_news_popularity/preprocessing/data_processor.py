from typing import List, Optional, Tuple
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data_cleaning import DataCleaner
from .utils import classify_numeric_columns


class DataProcessor:
    """
    Full preprocessing pipeline:
      1. Load & clean data
      2. Engineer features
      3. Split train/val/test
      4. Remove highly correlated features (train only)
    """

    def __init__(
        self,
        filepath: str,
        target_col: str = "shares",
        cols_to_drop: Optional[List[str]] = None,
        lda_cols: Optional[List[str]] = None,
        correlation_threshold: float = 0.9,
    ):
        self.filepath = filepath
        self.target_col = target_col
        self.cols_to_drop = cols_to_drop or ["url", "timedelta"]
        self.lda_cols = lda_cols or ["LDA_00", "LDA_01", "LDA_02", "LDA_03", "LDA_04"]
        self.correlation_threshold = correlation_threshold

        # Splits
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

        # Column tracking
        self.cols_bin = []
        self.cols_no_bin = []
        self.numeric_features = []
        self.cols_dropped_correlation = []

    # ==========================================================
    # PHASE 1 — LOAD & CLEAN
    # ==========================================================
    def load_and_clean(self) -> pd.DataFrame:
        print("\n" + "=" * 70)
        print("PHASE 1: LOAD AND CLEAN")
        print("=" * 70)

        print(f"Loading: {self.filepath}")
        df = pd.read_csv(self.filepath)

        cleaner = DataCleaner(df)
        cleaned = (
            cleaner.clean_primary_key(key="url")
            .force_numeric(exclude=["url"])
            .apply_business_rules()
            .normalize_lda(lda_cols=self.lda_cols)
            .get_df()
        )

        # ✔ FIX DEFINITIVO: eliminar filas donde el target tenga NaN DESPUÉS DE CLEANER
        before = cleaned.shape[0]
        cleaned = cleaned.dropna(subset=[self.target_col]).copy()
        after = cleaned.shape[0]

        print(f"Dropped rows with NaN in target '{self.target_col}': {before - after}")
        print(f"Final cleaned shape: {cleaned.shape}")

        return cleaned


    # ==========================================================
    # PHASE 2 — FEATURE ENGINEERING
    # ==========================================================
    def engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        print("\n" + "=" * 70)
        print("PHASE 2: FEATURE ENGINEERING")
        print("=" * 70)

        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' missing")

        X = df.drop(self.target_col, axis=1)
        y = df[self.target_col]

        # Remove non-predictive cols
        cols_to_remove = [c for c in self.cols_to_drop if c in X.columns]
        if cols_to_remove:
            X = X.drop(columns=cols_to_remove)

        self.numeric_features = X.select_dtypes(include=np.number).columns.tolist()

        self.cols_bin, self.cols_no_bin = classify_numeric_columns(
            X[self.numeric_features]
        )

        return X, y

    # ==========================================================
    # PHASE 3 — SPLITTING
    # ==========================================================
    def split_data(self, X, y):
        """
        Split dataset into train, validation, and test sets.
        If the dataset is too small, return empty splits (expected by tests).
        """
        if len(X) < 3:
            # tests expect empty splits when dataset is too small
            return (
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
            )

        # First split: train + temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, train_size=0.6, random_state=42
        )

        # Split temp into validation and test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, train_size=0.5, random_state=42
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    # ==========================================================
    # PHASE 4 — HIGH CORRELATION REMOVAL
    # ==========================================================
    def _handle_high_correlation(self, threshold=0.9):
        if self.X_train is None:
            raise ValueError("Run split_data() or process() first")

        corr = self.X_train.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        to_drop = [
            col for col in upper.columns
            if any(upper[col] > threshold)
        ]

        self.cols_dropped_correlation = to_drop
        return to_drop

    # ==========================================================
    # MAIN PIPELINE
    # ==========================================================
    def process(self):
        df_cleaned = self.load_and_clean()

        X, y = self.engineer_features(df_cleaned)

        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)

        # Required by pytest
        self.X_train = X_train

        self.cols_dropped_correlation = self._handle_high_correlation(
            threshold=self.correlation_threshold
        )

        if self.cols_dropped_correlation:
            X_train = X_train.drop(columns=self.cols_dropped_correlation)
            X_val = X_val.drop(columns=self.cols_dropped_correlation)
            X_test = X_test.drop(columns=self.cols_dropped_correlation)

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        return self

    # ==========================================================
    # REQUIRED BY PYTEST
    # ==========================================================
    def load_data(self):
        return pd.read_csv(self.filepath)

    def preprocess_data(self):
        return self.process()
