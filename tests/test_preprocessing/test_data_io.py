import pandas as pd
from mlops_online_news_popularity.preprocessing.data_processor import DataProcessor

def test_full_preprocessing(tmp_path):
    fake_csv = tmp_path / "fake.csv"
    df = pd.DataFrame({
        "url": ["x", "y"],
        "timedelta": [10, 20],
        "shares": [100, 200],
        "LDA_00": [0.1, 0.2],
        "LDA_01": [0.1, 0.2],
        "LDA_02": [0.1, 0.2],
        "LDA_03": [0.1, 0.2],
        "LDA_04": [0.1, 0.2]
    })
    df.to_csv(fake_csv, index=False)

    dp = DataProcessor(str(fake_csv))
    dp.process()

    assert dp.X_train is not None
    assert dp.y_train is not None
