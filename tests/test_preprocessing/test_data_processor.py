import pandas as pd
import numpy as np
from mlops_online_news_popularity.preprocessing.data_processor import DataProcessor

def test_filter_expected_columns():
    df = pd.DataFrame({
        "url": ["a"],
        "timedelta": [10],
        "extra": [123],
        "shares": [50]
    })
    dp = DataProcessor("dummy")
    dp.data = df

    dp.expected_cols = ["url", "timedelta", "shares"]

    dp.data = df
    result = dp.data[dp.expected_cols]

    assert list(result.columns) == ["url", "timedelta", "shares"]


def test_handle_high_correlation():
    dp = DataProcessor("dummy")

    dp.X_train = pd.DataFrame({
        "a": [1,2,3,4],
        "b": [2,4,6,8],  # correlación perfecta con a
        "c": [10,20,30,40]
    })
    dp.X_test = dp.X_train.copy()

    dp._handle_high_correlation(threshold=0.9)

    assert ("a" in dp.cols_to_drop) or ("b" in dp.cols_to_drop)
    assert dp.X_train.shape[1] == 2
