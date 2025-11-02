"""
Data I/O utilities for loading and saving data.

This module provides the DataLoader class for handling CSV file operations.
"""

import pandas as pd


class DataLoader:
    """
    Simple data loader for CSV files.

    This class provides utility methods for loading and saving CSV files
    with consistent formatting.
    """

    def load_csv(self, path):
        """
        Load a CSV file into a DataFrame.

        Parameters
        ----------
        path : str
            Path to the CSV file

        Returns
        -------
        pd.DataFrame
            Loaded DataFrame
        """
        return pd.read_csv(path)

    def save_csv(self, df, path):
        """
        Save a DataFrame to a CSV file.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to save
        path : str
            Path where to save the CSV file
        """
        df.to_csv(path, index=False)
        print(f"💾 Guardado en {path} (shape={df.shape})")
