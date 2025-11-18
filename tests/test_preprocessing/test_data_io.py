from mlops_online_news_popularity.preprocessing.data_io import DataLoader
import pandas as pd

def test_data_loader_loads_csv(tmp_path):
    csv_path = tmp_path / "sample.csv"
    df = pd.DataFrame({"a": [1,2,3]})
    df.to_csv(csv_path, index=False)

    loaded = DataLoader.load_csv(str(csv_path))

    assert loaded.shape == (3,1)
    assert "a" in loaded.columns
