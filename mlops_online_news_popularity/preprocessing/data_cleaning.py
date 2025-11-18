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

    def filter_expected_columns(self, expected_cols):
        """
        Filter DataFrame to only include expected columns.

        Parameters
        ----------
        expected_cols : list
            List of expected column names

        Returns
        -------
        self : DataCleaner
            Returns self for method chaining
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

    def force_numeric(self, exclude=["url"]):
        """
        Force columns to numeric type, handling common string representations.

        Parameters
        ----------
        exclude : list, optional
            Column names to exclude from numeric conversion

        Returns
        -------
        self : DataCleaner
            Returns self for method chaining
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

    def apply_business_rules(self):
        """
        Apply domain-specific business rules to the data.

        - Clips 'timedelta' to [0, 731] (2 years)
        - Clips proportions like 'n_unique_tokens' and 'global_subjectivity' to [0, 1]

        Returns
        -------
        self : DataCleaner
            Returns self for method chaining
        """
        print("Aplicando reglas de negocio...")
        # Clip timedelta to reasonable range (0 to 2 years)
        if "timedelta" in self.df:
            self.df["timedelta"] = self.df["timedelta"].clip(0, 731)

        # Clip proportion columns to [0, 1]
        clip_01 = ["n_unique_tokens", "global_subjectivity"]
        for c in clip_01:
            if c in self.df:
                self.df[c] = self.df[c].clip(0, 1)
        return self

    def winsorize_columns(self, exclude=set()):
        """
        Winsorize numeric columns to reduce impact of outliers.

        Clips values at 1st and 99th percentiles for each numeric column.

        Parameters
        ----------
        exclude : set, optional
            Set of column names to exclude from winsorization

        Returns
        -------
        self : DataCleaner
            Returns self for method chaining
        """
        print("Winsorizing columnas numéricas para controlar outliers...")

        def winsorize(s, low=0.01, high=0.99):
            if s.notna().sum() == 0:
                return s
            ql, qh = s.quantile(low), s.quantile(high)
            return s.clip(ql, qh)

        num_cols = [
            c for c in self.df.select_dtypes(include=[np.number]).columns if c not in exclude
        ]
        for c in num_cols:
            self.df[c] = winsorize(self.df[c])
        return self

    def normalize_lda(self, lda_cols=None):
        """
        Normalize LDA topic columns to sum to 1.0 per row.

        Parameters
        ----------
        lda_cols : list, optional
            List of LDA column names (e.g., ['LDA_00', 'LDA_01', ...])

        Returns
        -------
        self : DataCleaner
            Returns self for method chaining
        """
        if not lda_cols:
            return self

        print("Normalizando columnas LDA...")
        lda_cols = [c for c in lda_cols if c in self.df]
        if lda_cols:
            s = self.df[lda_cols].sum(axis=1)
            mask = s > 0
            self.df.loc[mask, lda_cols] = self.df.loc[mask, lda_cols].div(s[mask], axis=0)
        return self

    def clean_primary_key(self, key="url"):
        """
        Clean primary key column (typically 'url').

        - Removes null/empty values
        - Strips whitespace and converts to lowercase
        - Filters to only valid HTTP(S) URLs

        Parameters
        ----------
        key : str, optional
            Name of the primary key column (default: 'url')

        Returns
        -------
        self : DataCleaner
            Returns self for method chaining
        """
        print(f"Limpiando clave primaria '{key}'...")
        if key not in self.df.columns:
            print(f"Advertencia: Clave primaria '{key}' no encontrada para limpiar.")
            return self

        self.df = self.df[self.df[key].notna() & (self.df[key] != "")]
        self.df[key] = self.df[key].astype(str).str.strip().str.lower()
        self.df = self.df[self.df[key].str.startswith("http", na=False)]
        return self

    def impute_missing_values(self):
        """
        Impute missing values based on skewness.

        - For normally distributed columns (|skew| < 1): use mean
        - For skewed columns (|skew| >= 1): use median

        Returns
        -------
        self : DataCleaner
            Returns self for method chaining
        """
        print("Imputando valores faltantes basado en sesgo de distribución...")
        for col in self.df.columns[1:]:
            if self.df[col].isna().sum() == 0:
                continue

            skew = self.df[col].skew()
            if -1 < skew < 1:
                # Normal distribution: use mean
                val = self.df[col].mean()
                self.df[col] = self.df[col].fillna(val)
            else:
                # Skewed distribution: use median
                val = self.df[col].median()
                self.df[col] = self.df[col].fillna(val)
        return self

    def get_df(self):
        """
        Get the cleaned DataFrame.

        Returns
        -------
        pd.DataFrame
            The cleaned DataFrame
        """
        return self.df
