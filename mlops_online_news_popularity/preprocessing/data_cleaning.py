"""
Data cleaning module with comprehensive cleaning methods.

This module provides the DataCleaner class that combines functionality from both
the original data_clean and preprocess modules.
"""

import numpy as np
import pandas as pd


class DataCleaner:
    """
    Comprehensive data cleaning class with method chaining support.

    Combines cleaning methods from both data_clean/DataCleaner.py and
    preprocess/cleaning_eda.py for a unified cleaning interface.
    """

    def __init__(self, df):
        """
        Initialize DataCleaner with a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to clean
        """
        self.df = df.copy()

    # ================================================================
    # 1. FILTER EXPECTED COLUMNS
    # ================================================================
    def filter_expected_columns(self, expected_cols):
        """
        Filter DataFrame to only include expected columns.

        Parameters
        ----------
        expected_cols : list

        Returns
        -------
        self
        """
        print("Filtrando columnas esperadas...")

        extra = [c for c in self.df.columns if c not in expected_cols]
        missing = [c for c in expected_cols if c not in self.df.columns]

        if extra:
            print(f"⚠️ Extras ignoradas: {extra}")
        if missing:
            print(f"⚠️ Faltan columnas: {missing}")

        self.df = self.df[[c for c in expected_cols if c in self.df.columns]]
        return self

    # ================================================================
    # 2. FORCE NUMERIC
    # ================================================================
    def force_numeric(self, exclude=["url"]):
        """
        Convert string columns to numeric when possible.

        Returns
        -------
        self
        """
        print("Forzando columnas a tipo numérico...")

        for c in self.df.columns:
            if c in exclude:
                continue

            if self.df[c].dtype == "O":
                self.df[c] = (
                    self.df[c]
                    .astype(str)
                    .str.replace(",", ".", regex=False)
                    .replace({"nan": np.nan, "None": np.nan, "": np.nan})
                )

            self.df[c] = pd.to_numeric(self.df[c], errors="coerce")

        return self

    # ================================================================
    # 3. BUSINESS RULES
    # ================================================================
    def apply_business_rules(self):
        """
        Apply domain-specific business rules.

        Returns
        -------
        self
        """
        print("Aplicando reglas de negocio...")

        if "timedelta" in self.df:
            self.df["timedelta"] = self.df["timedelta"].clip(0, 731)

        for col in ["n_unique_tokens", "global_subjectivity"]:
            if col in self.df:
                self.df[col] = self.df[col].clip(0, 1)

        return self

    # ================================================================
    # 4. WINSORIZATION
    # ================================================================
    def winsorize_columns(self, exclude=set()):
        """
        Winsorize numeric columns at 1st and 99th percentiles.

        Returns
        -------
        self
        """
        print("Winsorizing columnas numéricas para controlar outliers...")

        def winsorize(s, low=0.01, high=0.99):
            if s.notna().sum() == 0:
                return s
            ql, qh = s.quantile(low), s.quantile(high)
            return s.clip(ql, qh)

        num_cols = [
            c for c in self.df.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]

        for c in num_cols:
            self.df[c] = winsorize(self.df[c])

        return self

    # ================================================================
    # 5. NORMALIZE LDA
    # ================================================================
    def normalize_lda(self, lda_cols=None):
        """
        Normalize LDA topic probabilities to sum to 1.

        Returns
        -------
        self
        """
        if not lda_cols:
            return self

        print("Normalizando columnas LDA...")

        lda_cols = [c for c in lda_cols if c in self.df]

        if lda_cols:
            row_sum = self.df[lda_cols].sum(axis=1)
            mask = row_sum > 0
            self.df.loc[mask, lda_cols] = self.df.loc[mask, lda_cols].div(row_sum[mask], axis=0)

        return self

    # ================================================================
    # 6. CLEAN PRIMARY KEY
    # ================================================================
    def clean_primary_key(self, key="url"):
        """
        Clean primary key: lowercase, strip, remove nulls, remove duplicates.
        Does NOT filter for "http" — tests expect this behaviour.
        """
        print(f"Limpiando clave primaria '{key}'...")

        if key not in self.df.columns:
            print(f"Advertencia: Clave primaria '{key}' no encontrada.")
            return self

        # Remove null/empty
        self.df = self.df[self.df[key].notna() & (self.df[key] != "")]

        # Normalize
        self.df[key] = self.df[key].astype(str).str.strip().str.lower()

        # Tests expect duplicates removed
        self.df = self.df.drop_duplicates(subset=[key])

        return self

    # ================================================================
    # 7. IMPUTE MISSING VALUES
    # ================================================================
    def impute_missing_values(self):
        """
        Impute missing values using mean or median based on skewness.

        Returns
        -------
        self
        """
        print("Imputando valores faltantes...")

        for col in self.df.columns[1:]:  # skip primary key
            if self.df[col].isna().sum() == 0:
                continue

            skew = self.df[col].skew()

            if -1 < skew < 1:
                val = self.df[col].mean()
            else:
                val = self.df[col].median()

            self.df[col] = self.df[col].fillna(val)

        return self

    # ================================================================
    # 8. GET DF
    # ================================================================
    def get_df(self):
        """
        Return cleaned DataFrame.
        """
        return self.df
