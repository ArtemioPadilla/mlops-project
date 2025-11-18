import pandas as pd
from mlops_online_news_popularity.preprocessing.data_processor import DataProcessor

def test_load_and_clean_removes_nan_target(tmp_path):
    fake_csv = tmp_path / "fake.csv"
    fake_data = pd.DataFrame({
        "url": ["http://example.com/a", "http://example.com/b", "http://example.com/c"],
        "timedelta": [1, 2, 3],
        "LDA_00": [0.1, 0.2, 0.3],
        "LDA_01": [0.1, 0.2, 0.3],
        "LDA_02": [0.1, 0.2, 0.3],
        "LDA_03": [0.1, 0.2, 0.3],
        "LDA_04": [0.1, 0.2, 0.3],
        "shares": [100, None, 300]
    })
    fake_data.to_csv(fake_csv, index=False)

    dp = DataProcessor(filepath=str(fake_csv))
    df = dp.load_and_clean()

    # No debe haber NaN en target
    assert df["shares"].isna().sum() == 0
    assert df.shape[0] == 2  # Se eliminó 1 fila
