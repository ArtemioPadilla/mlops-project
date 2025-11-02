"""
Utility functions for data preprocessing.

This module provides helper functions for data preprocessing tasks.
"""

import pandas as pd


def classify_numeric_columns(df_numeric: pd.DataFrame) -> tuple[list, list]:
    """
    Classify numeric columns as binary or non-binary.

    Binary columns are those that only contain values 0 and 1.

    Parameters
    ----------
    df_numeric : pd.DataFrame
        DataFrame with numeric columns

    Returns
    -------
    tuple[list, list]
        Tuple of (binary_columns, non_binary_columns)

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': [0, 1, 0, 1], 'b': [1.2, 3.4, 5.6, 7.8]})
    >>> binary, non_binary = classify_numeric_columns(df)
    >>> binary
    ['a']
    >>> non_binary
    ['b']
    """
    cols_bin = [
        col for col in df_numeric.columns if set(df_numeric[col].dropna().unique()) <= {0, 1}
    ]
    cols_no_bin = [col for col in df_numeric.columns if col not in cols_bin]

    print(f"Columnas binarias identificadas: {len(cols_bin)}")
    print(f"Columnas numéricas no binarias: {len(cols_no_bin)}")

    return cols_bin, cols_no_bin
