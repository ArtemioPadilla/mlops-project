import pandas as pd
from mlops_online_news_popularity.preprocessing.data_processor import DataProcessor
from mlops_online_news_popularity.modeling.train import train_model
from mlops_online_news_popularity.modeling.predict import load_model, predict
import os

def test_full_pipeline(tmp_path):
    # 1. Crear dataset sintético
    df = pd.DataFrame({
        "url": ["http://a.com"] * 20,
        "timedelta": [10] * 20,
        "n_tokens_title": [5]*20,
        "n_tokens_content": [100]*20,
        "n_unique_tokens": [0.5]*20,
        "n_non_stop_words": [0.8]*20,
        "n_non_stop_unique_tokens": [0.9]*20,
        "num_hrefs": [3]*20,
        "num_self_hrefs": [1]*20,
        "num_imgs": [1]*20,
        "num_videos": [0]*20,
        "average_token_length": [4]*20,
        "num_keywords": [5]*20,
        "data_channel_is_lifestyle": [0]*20,
        "data_channel_is_entertainment": [0]*20,
        "data_channel_is_bus": [1]*20,
        "data_channel_is_socmed": [0]*20,
        "data_channel_is_tech": [0]*20,
        "data_channel_is_world": [0]*20,
        "kw_min_min": [1]*20,
        "kw_max_min": [2]*20,
        "kw_avg_min": [1.5]*20,
        "kw_min_max": [5]*20,
        "kw_max_max": [20]*20,
        "kw_avg_max": [10]*20,
        "kw_min_avg": [2]*20,
        "kw_max_avg": [8]*20,
        "kw_avg_avg": [5]*20,
        "self_reference_min_shares": [100]*20,
        "self_reference_max_shares": [200]*20,
        "self_reference_avg_sharess": [150]*20,
        "weekday_is_monday": [1]*20,
        "weekday_is_tuesday": [0]*20,
        "weekday_is_wednesday": [0]*20,
        "weekday_is_thursday": [0]*20,
        "weekday_is_friday": [0]*20,
        "weekday_is_saturday": [0]*20,
        "weekday_is_sunday": [0]*20,
        "is_weekend": [0]*20,
        "LDA_00": [0.2]*20,
        "LDA_01": [0.2]*20,
        "LDA_02": [0.2]*20,
        "LDA_03": [0.2]*20,
        "LDA_04": [0.2]*20,
        "global_subjectivity": [0.4]*20,
        "global_sentiment_polarity": [0.1]*20,
        "global_rate_positive_words": [0.05]*20,
        "global_rate_negative_words": [0.03]*20,
        "rate_positive_words": [0.04]*20,
        "rate_negative_words": [0.02]*20,
        "avg_positive_polarity": [0.3]*20,
        "min_positive_polarity": [0.1]*20,
        "max_positive_polarity": [0.5]*20,
        "avg_negative_polarity": [-0.2]*20,
        "min_negative_polarity": [-0.5]*20,
        "max_negative_polarity": [-0.1]*20,
        "title_subjectivity": [0.2]*20,
        "title_sentiment_polarity": [0.0]*20,
        "abs_title_subjectivity": [0.2]*20,
        "abs_title_sentiment_polarity": [0.1]*20,
        "shares": [1500]*20
    })

    csv_path = tmp_path / "synthetic.csv"
    df.to_csv(csv_path, index=False)

    # 2. Procesamiento
    dp = DataProcessor(str(csv_path))
    dp.load_data()
    dp.preprocess_data()

    assert dp.X_train is not None
    assert dp.X_test is not None

    # 3. Entrenamiento
    model_path = train_model(dp.X_train, dp.y_train, tmp_path)
    assert os.path.exists(model_path)

    # 4. Predicción
    model = load_model(model_path)
    y_pred = predict(model, dp.X_test.iloc[[0]])

    assert y_pred in [0,1]
