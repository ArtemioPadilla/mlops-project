"""
Pytest fixtures for serving module tests.

Avoid importing FastAPI or config at module import time,
because pytest will import this file BEFORE running tests.
"""

import tempfile
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient


# 🚨 IMPORTANT: NO FastAPI imports here
# NO: from mlops_online_news_popularity.serving import config
# NO: from mlops_online_news_popularity.serving.app import app
# These must only be imported inside fixtures!


@pytest.fixture
def sample_features() -> Dict[str, float]:
    """
    Sample news article features for testing.

    Returns a dictionary with all 59 required features for model prediction.
    """
    return {
        "n_tokens_title": 10.0,
        "n_tokens_content": 500.0,
        "n_unique_tokens": 0.5,
        "n_non_stop_words": 0.8,
        "n_non_stop_unique_tokens": 0.6,
        "num_hrefs": 10.0,
        "num_self_hrefs": 2.0,
        "num_imgs": 5.0,
        "num_videos": 1.0,
        "average_token_length": 4.5,
        "num_keywords": 7.0,
        "data_channel_is_lifestyle": 0.0,
        "data_channel_is_entertainment": 0.0,
        "data_channel_is_bus": 1.0,
        "data_channel_is_socmed": 0.0,
        "data_channel_is_tech": 0.0,
        "data_channel_is_world": 0.0,
        "kw_min_min": 0.0,
        "kw_max_min": 1000.0,
        "kw_avg_min": 300.0,
        "kw_min_max": 0.0,
        "kw_max_max": 50000.0,
        "kw_avg_max": 10000.0,
        "kw_min_avg": 0.0,
        "kw_max_avg": 5000.0,
        "kw_avg_avg": 2500.0,
        "self_reference_min_shares": 1000.0,
        "self_reference_max_shares": 10000.0,
        "self_reference_avg_sharess": 5000.0,
        "weekday_is_monday": 0.0,
        "weekday_is_tuesday": 1.0,
        "weekday_is_wednesday": 0.0,
        "weekday_is_thursday": 0.0,
        "weekday_is_friday": 0.0,
        "weekday_is_saturday": 0.0,
        "weekday_is_sunday": 0.0,
        "is_weekend": 0.0,
        "LDA_00": 0.2,
        "LDA_01": 0.3,
        "LDA_02": 0.2,
        "LDA_03": 0.2,
        "LDA_04": 0.1,
        "global_subjectivity": 0.5,
        "global_sentiment_polarity": 0.1,
        "global_rate_positive_words": 0.04,
        "global_rate_negative_words": 0.02,
        "rate_positive_words": 0.7,
        "rate_negative_words": 0.3,
        "avg_positive_polarity": 0.35,
        "min_positive_polarity": 0.1,
        "max_positive_polarity": 1.0,
        "avg_negative_polarity": -0.25,
        "min_negative_polarity": -0.8,
        "max_negative_polarity": -0.05,
        "title_subjectivity": 0.5,
        "title_sentiment_polarity": 0.0,
        "abs_title_subjectivity": 0.0,
        "abs_title_sentiment_polarity": 0.0,
        "mixed_type_col": 0.0,
    }


@pytest.fixture
def mock_sklearn_pipeline():
    """
    Create a simple real sklearn Pipeline for testing.

    Returns a minimal but functional sklearn pipeline that can be pickled.
    Uses a Ridge regression model fitted with dummy data.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    # Create minimal real pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0, random_state=42))
    ])

    # Fit with dummy data (100 samples, 59 features to match real model)
    np.random.seed(42)
    X_dummy = np.random.rand(100, 59)
    # Create y_dummy with log-transformed values around 7.824 (log1p(2500))
    y_dummy = np.random.normal(7.824, 0.5, 100)
    pipeline.fit(X_dummy, y_dummy)

    return pipeline


@pytest.fixture
def temp_model_file(mock_sklearn_pipeline, tmp_path):
    """
    Create a temporary model file for testing.

    Args:
        mock_sklearn_pipeline: Mock model fixture
        tmp_path: pytest temporary directory

    Returns:
        Path to the temporary model file
    """
    model_path = tmp_path / "test_model.pkl"
    joblib.dump(mock_sklearn_pipeline, model_path)
    return model_path


@pytest.fixture
def test_client():
    """
    Create a FastAPI TestClient for integration tests.

    Note: This imports the app, which will try to load a model on startup.
    The model loading may fail in test environment, which is expected.
    Use `raise_server_exceptions=False` to prevent startup errors from failing tests.
    """
    from mlops_online_news_popularity.serving.app import app
    # Create client without triggering startup event exceptions
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def initialized_test_client(mock_sklearn_pipeline, temp_model_file, monkeypatch):
    """
    Create a FastAPI TestClient with initialized model.

    This fixture properly initializes the model handler with a mock model
    so all endpoints work correctly. The model handler is reset after the test.

    Args:
        mock_sklearn_pipeline: Mock sklearn pipeline fixture
        temp_model_file: Temporary model file path fixture
        monkeypatch: pytest monkeypatch fixture for environment variables
    """
    from mlops_online_news_popularity.serving.app import app
    from mlops_online_news_popularity.serving.model_handler import get_model_handler

    # Get handler first and reset any previous state
    handler = get_model_handler()
    handler.initialized = False
    handler.model = None
    # Clear any cached attributes that might reference old paths
    if hasattr(handler, 'model_path'):
        handler.model_path = None
    if hasattr(handler, '_model_info'):
        handler._model_info = None

    # NOW set environment variables (after reset)
    monkeypatch.setenv("MODEL_LOAD_STRATEGY", "local")
    monkeypatch.setenv("MODEL_PATH", str(temp_model_file))

    # Initialize handler with the new environment
    handler.initialize()

    # Create test client
    with TestClient(app) as client:
        yield client

    # Cleanup - reset handler state
    handler.initialized = False
    handler.model = None


@pytest.fixture
def sample_csv_content() -> bytes:
    """
    Sample CSV content for file upload testing.

    Returns CSV bytes with header and 2 sample rows containing all 59 features.
    """
    csv_data = """n_tokens_title,n_tokens_content,n_unique_tokens,n_non_stop_words,n_non_stop_unique_tokens,num_hrefs,num_self_hrefs,num_imgs,num_videos,average_token_length,num_keywords,data_channel_is_lifestyle,data_channel_is_entertainment,data_channel_is_bus,data_channel_is_socmed,data_channel_is_tech,data_channel_is_world,kw_min_min,kw_max_min,kw_avg_min,kw_min_max,kw_max_max,kw_avg_max,kw_min_avg,kw_max_avg,kw_avg_avg,self_reference_min_shares,self_reference_max_shares,self_reference_avg_sharess,weekday_is_monday,weekday_is_tuesday,weekday_is_wednesday,weekday_is_thursday,weekday_is_friday,weekday_is_saturday,weekday_is_sunday,is_weekend,LDA_00,LDA_01,LDA_02,LDA_03,LDA_04,global_subjectivity,global_sentiment_polarity,global_rate_positive_words,global_rate_negative_words,rate_positive_words,rate_negative_words,avg_positive_polarity,min_positive_polarity,max_positive_polarity,avg_negative_polarity,min_negative_polarity,max_negative_polarity,title_subjectivity,title_sentiment_polarity,abs_title_subjectivity,abs_title_sentiment_polarity,mixed_type_col
10.0,500.0,0.5,0.8,0.6,10.0,2.0,5.0,1.0,4.5,7.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1000.0,300.0,0.0,50000.0,10000.0,0.0,5000.0,2500.0,1000.0,10000.0,5000.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.2,0.3,0.2,0.2,0.1,0.5,0.1,0.04,0.02,0.7,0.3,0.35,0.1,1.0,-0.25,-0.8,-0.05,0.5,0.0,0.0,0.0,0.0
8.0,300.0,0.6,0.75,0.55,8.0,1.0,3.0,0.0,4.2,6.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,800.0,250.0,0.0,40000.0,8000.0,0.0,4000.0,2000.0,800.0,8000.0,4000.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.25,0.25,0.25,0.15,0.1,0.45,0.05,0.03,0.025,0.65,0.35,0.3,0.08,0.9,-0.3,-0.85,-0.1,0.4,0.02,0.1,0.02,0.0"""
    return csv_data.encode("utf-8")
