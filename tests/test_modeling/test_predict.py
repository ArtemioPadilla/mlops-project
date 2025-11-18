import pandas as pd
from mlops_online_news_popularity.modeling.predict import predict

def test_predict_output_is_list(mock_model):
    X = pd.DataFrame([[0.1] * len(mock_model.feature_names_in_)],
                     columns=mock_model.feature_names_in_)

    result = predict(mock_model, X)
    
    assert isinstance(result, (list, tuple)), "predict() debe regresar lista/tuple"
    assert len(result) == 1

def test_predict_adds_missing_columns(mock_model):
    # X con columnas incompletas
    partial = pd.DataFrame([{}])

    result = predict(mock_model, partial)

    # Debe completar las columnas faltantes con 0
    assert len(result) == 1
