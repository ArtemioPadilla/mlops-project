import pandas as pd
from mlops_online_news_popularity.preprocessing.data_cleaning import DataCleaner

def test_primary_key_cleaning():
    df = pd.DataFrame({
        "url": ["http://example.com/a", "http://example.com/a", "http://example.com/b"],
        "value": [1, 2, 3]
    })
    cleaner = DataCleaner(df)
    df_clean = cleaner.clean_primary_key("url").get_df()

    # Después de limpiar, deben quedar solo URLs válidas (las 3 son válidas)
    assert len(df_clean) == 3
    # Las URLs se convierten a lowercase
    assert set(df_clean["url"]) == {"http://example.com/a", "http://example.com/b"}
