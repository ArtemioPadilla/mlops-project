import pandas as pd
from mlops_online_news_popularity.preprocessing.data_cleaning import DataCleaner

def test_clean_primary_key():
    df = pd.DataFrame({
        "url": ["http://valid.com", "invalid", None, ""],
        "n_tokens_title": [10, 20, 30, 40]
    })

    cleaner = DataCleaner(df).clean_primary_key("url")
    result = cleaner.get_df()

    assert len(result) == 1
    assert result.iloc[0]["url"].startswith("http")

def test_force_numeric():
    df = pd.DataFrame({
        "num": ["10", "5.5", "nan", None, "abc"],
        "url": ["x", "y", "z", "a", "b"]
    })
    cleaner = DataCleaner(df).force_numeric()
    result = cleaner.get_df()

    assert result["num"].dtype != object
    assert result["num"].isna().sum() >= 2

def test_apply_business_rules():
    df = pd.DataFrame({
        "timedelta": [1000, -5, 30],
        "n_unique_tokens": [1.5, -0.1, 0.5]
    })

    cleaner = DataCleaner(df).apply_business_rules()
    result = cleaner.get_df()

    assert result["timedelta"].max() <= 731
    assert result["n_unique_tokens"].between(0, 1).all()

