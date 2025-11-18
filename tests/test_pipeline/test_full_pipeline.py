from mlops_online_news_popularity.preprocessing.data_processor import DataProcessor
from mlops_online_news_popularity.modeling.train import train_model
from mlops_online_news_popularity.modeling.predict import predict
import pandas as pd
import os

def test_full_pipeline_e2e(tmp_path):
    # 1) Crear dataset simple con URLs válidas
    csv = tmp_path / "data.csv"
    df = pd.DataFrame({
        "url": [
            "http://example.com/a",
            "http://example.com/b",
            "http://example.com/c",
            "http://example.com/d"
        ],
        "timedelta": [1, 2, 3, 4],
        "LDA_00": [0.1] * 4,
        "LDA_01": [0.2] * 4,
        "LDA_02": [0.3] * 4,
        "LDA_03": [0.4] * 4,
        "LDA_04": [0.5] * 4,
        "shares": [100, 120, 130, 90]
    })
    df.to_csv(csv, index=False)

    # 2) Preprocesar
    dp = DataProcessor(str(csv))
    dp.process()

    # 3) Entrenar modelo
    model = train_model(dp.X_train, dp.y_train)

    # 4) Verificar inferencia
    preds = predict(model, dp.X_test)

    assert len(preds) == len(dp.X_test)
    assert all([p > 0 for p in preds])
