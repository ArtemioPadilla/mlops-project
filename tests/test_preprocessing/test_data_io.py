import pandas as pd
from mlops_online_news_popularity.preprocessing.data_processor import DataProcessor

def test_full_preprocessing(tmp_path):
    fake_csv = tmp_path / "fake.csv"
    # Need at least 10 rows for 70/15/15 split to work properly
    df = pd.DataFrame({
        "url": [f"http://example.com/{i}" for i in range(10)],
        "timedelta": [10 + i for i in range(10)],
        "shares": [100 + i * 10 for i in range(10)],
        "LDA_00": [0.1 + i * 0.01 for i in range(10)],
        "LDA_01": [0.2 + i * 0.01 for i in range(10)],
        "LDA_02": [0.3 + i * 0.01 for i in range(10)],
        "LDA_03": [0.4 + i * 0.01 for i in range(10)],
        "LDA_04": [0.1 - i * 0.01 for i in range(10)]
    })
    df.to_csv(fake_csv, index=False)

    dp = DataProcessor(str(fake_csv))
    dp.process()

    assert dp.X_train is not None
    assert dp.y_train is not None
    assert len(dp.X_train) > 0
    assert len(dp.X_val) > 0
    assert len(dp.X_test) > 0
