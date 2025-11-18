from mlops_online_news_popularity.modeling.predict import load_model, predict
import pandas as pd

def test_model_prediction():
    # Cargar modelo real entrenado
    model = load_model("models/model.pkl")

    # Crear input simple
    X = pd.DataFrame([{
        "n_tokens_title": 12,
        "kw_avg_min": 5,
        "global_subjectivity": 0.5,
        "LDA_00": 0.1, "LDA_01": 0.2, "LDA_02": 0.3, "LDA_03": 0.4, "LDA_04": 0.5
    }])

    result = predict(model, X)

    assert isinstance(result, list)
    assert len(result) == 1
