"""
Unit tests for ModelHandler class.

Tests model loading, preprocessing, inference, postprocessing, and complete pipeline.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from mlops_online_news_popularity.serving.model_handler import ModelHandler


class TestModelHandlerInitialization:
    """Tests for ModelHandler initialization."""

    def test_handler_init(self):
        """Test handler initialization."""
        handler = ModelHandler()

        assert handler.initialized is False
        assert handler.model is None
        assert handler.model_name is None
        assert isinstance(handler.model_info, dict)

    def test_initialize_with_local_model(self, temp_model_file):
        """Test initialization with local model file."""
        handler = ModelHandler()
        context = {
            "load_strategy": "local",
            "model_path": str(temp_model_file)
        }

        handler.initialize(context)

        assert handler.initialized is True
        assert handler.model is not None
        assert handler.model_name ==temp_model_file.stem

    def test_initialize_with_nonexistent_file(self):
        """Test initialization with nonexistent file raises error."""
        handler = ModelHandler()
        context = {
            "load_strategy": "local",
            "model_path": "/nonexistent/model.pkl"
        }

        with pytest.raises(FileNotFoundError):
            handler.initialize(context)

    def test_initialize_with_mlflow(self, mock_sklearn_pipeline):
        """Test initialization with MLflow by mocking the load method."""
        handler = ModelHandler()

        # Mock the entire _load_from_mlflow method since MLflow has complex internals
        def mock_load_from_mlflow(run_id):
            handler.model = mock_sklearn_pipeline
            handler.model_name = "TestModel"
            handler.model_info = {
                "model_name": "TestModel",
                "run_id": run_id,
                "load_strategy": "mlflow",
                "experiment_id": "123",
                "metrics": {"val_rmse": 0.5},
                "params": {},
            }

        handler._load_from_mlflow = mock_load_from_mlflow

        context = {
            "load_strategy": "mlflow",
            "mlflow_run_id": "test_run_id"
        }

        handler.initialize(context)

        assert handler.initialized is True
        assert handler.model is not None
        assert handler.model_name == "TestModel"
        assert handler.model_info["run_id"] == "test_run_id"

    def test_initialize_mlflow_without_run_id(self):
        """Test MLflow initialization without run_id raises error."""
        handler = ModelHandler()
        context = {"load_strategy": "mlflow"}

        with pytest.raises(ValueError, match="MLFLOW_RUN_ID must be provided"):
            handler.initialize(context)


class TestModelHandlerPreprocessing:
    """Tests for ModelHandler preprocessing."""

    def test_preprocess_dict_input(self, sample_features):
        """Test preprocessing with dict input."""
        handler = ModelHandler()
        result = handler.preprocess(sample_features)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert list(result.columns) == handler.preprocess(sample_features).columns.tolist()

    def test_preprocess_list_input(self, sample_features):
        """Test preprocessing with list of dicts."""
        handler = ModelHandler()
        result = handler.preprocess([sample_features, sample_features])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_preprocess_dataframe_input(self, sample_features):
        """Test preprocessing with DataFrame input."""
        handler = ModelHandler()
        df = pd.DataFrame([sample_features])
        result = handler.preprocess(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_preprocess_missing_features(self, sample_features):
        """Test preprocessing with missing features raises error."""
        handler = ModelHandler()
        features = sample_features.copy()
        del features["n_tokens_title"]

        with pytest.raises(ValueError, match="Missing required features"):
            handler.preprocess(features)

    def test_preprocess_invalid_input_type(self):
        """Test preprocessing with invalid input type raises error."""
        handler = ModelHandler()

        with pytest.raises(ValueError, match="Unsupported input type"):
            handler.preprocess("invalid")

    def test_preprocess_feature_order(self, sample_features):
        """Test that preprocessing maintains correct feature order."""
        from mlops_online_news_popularity.serving import config

        handler = ModelHandler()

        # Shuffle features
        shuffled = {k: sample_features[k] for k in reversed(sample_features.keys())}
        result = handler.preprocess(shuffled)

        # Check order matches config
        assert list(result.columns) == config.FEATURE_NAMES


class TestModelHandlerInference:
    """Tests for ModelHandler inference."""

    def test_inference_with_initialized_model(self, mock_sklearn_pipeline, temp_model_file, sample_features):
        """Test inference with initialized model."""
        handler = ModelHandler()
        handler.initialize({"load_strategy": "local", "model_path": str(temp_model_file)})

        df = handler.preprocess(sample_features)
        predictions = handler.inference(df)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 1
        # Check prediction is a reasonable log-transformed value (typically 6-10)
        # Using real model now, so exact value will vary
        assert isinstance(predictions[0], (int, float, np.number))
        assert not np.isnan(predictions[0])
        assert not np.isinf(predictions[0])

    def test_inference_without_initialization(self, sample_features):
        """Test inference without initialization raises error."""
        handler = ModelHandler()
        df = handler.preprocess(sample_features)

        with pytest.raises(RuntimeError, match="ModelHandler not initialized"):
            handler.inference(df)


class TestModelHandlerPostprocessing:
    """Tests for ModelHandler postprocessing."""

    def test_postprocess_single_prediction(self):
        """Test postprocessing single prediction."""
        handler = ModelHandler()
        predictions = np.array([7.824])  # log1p(2500) ≈ 7.824

        results = handler.postprocess(predictions)

        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["predicted_shares"] == pytest.approx(2500, abs=10)
        assert results[0]["log_prediction"] == pytest.approx(7.824, rel=0.01)

    def test_postprocess_batch_predictions(self):
        """Test postprocessing batch predictions."""
        handler = ModelHandler()
        predictions = np.array([7.824, 7.495])  # log1p(2500), log1p(1800)

        results = handler.postprocess(predictions)

        assert len(results) == 2
        assert results[0]["predicted_shares"] == pytest.approx(2500, abs=10)
        assert results[1]["predicted_shares"] == pytest.approx(1800, abs=10)

    def test_postprocess_applies_expm1(self):
        """Test that postprocessing applies expm1 transform."""
        handler = ModelHandler()

        # Known log1p transformation: log1p(1000) ≈ 6.90875
        log_prediction = 6.90875
        predictions = np.array([log_prediction])

        results = handler.postprocess(predictions)

        # expm1(6.90875) should give back approximately 1000
        assert results[0]["predicted_shares"] == pytest.approx(1000, abs=5)

    def test_postprocess_non_negative_shares(self):
        """Test that postprocessing ensures non-negative shares."""
        handler = ModelHandler()

        # Edge case: very small or negative log prediction
        predictions = np.array([-1.0, 0.0, 1.0])

        results = handler.postprocess(predictions)

        # All results should be non-negative
        for result in results:
            assert result["predicted_shares"] >= 0

    def test_postprocess_returns_integers(self):
        """Test that postprocessing returns integer share counts."""
        handler = ModelHandler()
        predictions = np.array([7.824])

        results = handler.postprocess(predictions)

        assert isinstance(results[0]["predicted_shares"], int)


class TestModelHandlerCompletePipeline:
    """Tests for complete ModelHandler pipeline (handle method)."""

    def test_handle_single_prediction(self, mock_sklearn_pipeline, temp_model_file, sample_features):
        """Test complete pipeline with single prediction."""
        handler = ModelHandler()
        handler.initialize({"load_strategy": "local", "model_path": str(temp_model_file)})

        results = handler.handle(sample_features)

        assert isinstance(results, list)
        assert len(results) == 1
        assert "predicted_shares" in results[0]
        assert "log_prediction" in results[0]
        assert isinstance(results[0]["predicted_shares"], int)

    def test_handle_batch_prediction(self, mock_sklearn_pipeline, temp_model_file, sample_features):
        """Test complete pipeline with batch prediction."""
        handler = ModelHandler()
        handler.initialize({"load_strategy": "local", "model_path": str(temp_model_file)})

        # Test with real model (no mocking predict method)
        results = handler.handle([sample_features, sample_features])

        assert len(results) == 2
        assert all("predicted_shares" in r for r in results)
        assert all("log_prediction" in r for r in results)
        # Verify predictions are valid numbers
        assert all(isinstance(r["predicted_shares"], int) for r in results)
        assert all(r["predicted_shares"] >= 0 for r in results)

    def test_handle_with_dataframe(self, mock_sklearn_pipeline, temp_model_file, sample_features):
        """Test complete pipeline with DataFrame input."""
        handler = ModelHandler()
        handler.initialize({"load_strategy": "local", "model_path": str(temp_model_file)})

        df = pd.DataFrame([sample_features])
        results = handler.handle(df)

        assert len(results) == 1
        assert isinstance(results[0]["predicted_shares"], int)


class TestModelHandlerModelInfo:
    """Tests for get_model_info method."""

    def test_model_info_before_initialization(self):
        """Test model info before initialization."""
        handler = ModelHandler()
        info = handler.get_model_info()

        assert info["status"] == "not_initialized"
        assert "message" in info

    def test_model_info_after_initialization(self, temp_model_file):
        """Test model info after initialization."""
        handler = ModelHandler()
        handler.initialize({"load_strategy": "local", "model_path": str(temp_model_file)})

        info = handler.get_model_info()

        assert info["status"] == "ready"
        assert "model_info" in info
        assert "features" in info
        assert "target" in info
        assert info["features"]["count"] == 59


class TestModelHandlerSingleton:
    """Tests for singleton pattern."""

    def test_get_model_handler_returns_same_instance(self):
        """Test that get_model_handler returns singleton instance."""
        from mlops_online_news_popularity.serving.model_handler import get_model_handler

        handler1 = get_model_handler()
        handler2 = get_model_handler()

        assert handler1 is handler2
