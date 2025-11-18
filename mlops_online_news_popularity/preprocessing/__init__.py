"""
Preprocessing module for data cleaning, exploration, and preparation.

This module provides comprehensive tools for:
- Data processing pipeline (DataProcessor)
- Data cleaning (DataCleaner)
- Exploratory data analysis (DataExplorer)
- Data loading/saving (DataLoader)
- Dataset comparison (DataComparator)
- Utility functions (classify_numeric_columns)
"""

from .data_cleaning import DataCleaner
from .data_comparison import DataComparator
from .data_exploration import DataExplorer
from .data_io import DataLoader
from .data_processor import DataProcessor
from .utils import classify_numeric_columns

__all__ = [
    "DataProcessor",
    "DataCleaner",
    "DataExplorer",
    "DataLoader",
    "DataComparator",
    "classify_numeric_columns",
]
