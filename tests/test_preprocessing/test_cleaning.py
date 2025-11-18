import pandas as pd
from mlops_online_news_popularity.preprocessing.data_cleaning import DataCleaner

def test_primary_key_cleaning():
    df = pd.DataFrame({"url": ["a", "a", "b"], "value": [1, 2, 3]})
    cleaner = DataCleaner(df)
    df_clean = cleaner.clean_primary_key("url").get_df()

    # Valores duplicados deben quedarse con el último
    assert len(df_clean) == 2
    assert set(df_clean["url"]) == {"a", "b"}
