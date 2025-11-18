"""
Preprocessing module for data cleaning, exploration, and preparation.

Used by the unit tests in:
- tests/test_preprocessing/
- tests/test_pipeline/

This module provides:
- Data processing pipeline (DataProcessor)
- Data cleaning operations (DataCleaner)
- Exploratory data analysis (DataExplorer)
- Data loading/saving utilities (DataLoader)
- Dataset comparison tools (DataComparator)
- Helper utilities (classify_numeric_columns)
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
