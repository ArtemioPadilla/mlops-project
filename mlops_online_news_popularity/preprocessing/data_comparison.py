"""
Data comparison utilities for comparing datasets.

This module provides the DataComparator class for comparing original and cleaned datasets.
"""

import pandas as pd


class DataComparator:
    """
    Compare statistics between original and cleaned datasets.

    This class provides methods for generating comparison reports between two datasets,
    typically used to compare original data with cleaned data.
    """

    def __init__(self, orig, clean):
        """
        Initialize DataComparator with original and cleaned datasets.

        Parameters
        ----------
        orig : pd.DataFrame
            Original dataset
        clean : pd.DataFrame
            Cleaned dataset
        """
        self.orig = orig
        self.clean = clean
        self.report = pd.DataFrame()  # Initialize empty to avoid None issues

    def compare_stats(self):
        """
        Calculate descriptive statistics (mean and median) for both datasets.

        Returns
        -------
        self : DataComparator
            Returns self for method chaining
        """
        self.report = pd.DataFrame(
            {
                "mean_orig": self.orig.mean(numeric_only=True),
                "mean_clean": self.clean.mean(numeric_only=True),
                "median_orig": self.orig.median(numeric_only=True),
                "median_clean": self.clean.median(numeric_only=True),
            }
        )
        return self

    def add_differences(self):
        """
        Add absolute differences between original and cleaned statistics.

        Returns
        -------
        self : DataComparator
            Returns self for method chaining

        Raises
        ------
        ValueError
            If compare_stats() hasn't been called first
        """
        if self.report.empty:
            raise ValueError("Primero ejecuta compare_stats() antes de add_differences().")
        self.report["diff_mean"] = (self.report["mean_clean"] - self.report["mean_orig"]).abs()
        self.report["diff_median"] = (
            self.report["median_clean"] - self.report["median_orig"]
        ).abs()
        return self

    def missing_values_ratio(self):
        """
        Calculate missing values ratio in percentage for both datasets.

        Returns
        -------
        self : DataComparator
            Returns self for method chaining
        """
        self.report["missing_orig_%"] = (self.orig.isna().sum() / len(self.orig)) * 100
        self.report["missing_clean_%"] = (self.clean.isna().sum() / len(self.clean)) * 100
        return self

    def export_report(self, path):
        """
        Export the comparison report to a CSV file.

        Parameters
        ----------
        path : str
            Path where to save the report

        Returns
        -------
        pd.DataFrame
            The comparison report

        Raises
        ------
        ValueError
            If no report has been generated yet
        """
        if self.report.empty:
            raise ValueError("No hay reporte que exportar. Ejecuta los métodos primero.")
        self.report.to_csv(path, index=False)
        print(f"📊 Reporte exportado a {path}")
        return self.report
