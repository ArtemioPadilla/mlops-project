"""
Unit tests for Pydantic schemas.

Tests validation, serialization, and schema definitions for all API models.
"""

import pytest
from pydantic import ValidationError

from mlops_online_news_popularity.serving import schemas


class TestNewsArticleFeatures:
    """Tests for NewsArticleFeatures schema."""

    def test_valid_features(self, sample_features):
        """Test that valid features are accepted."""
        article = schemas.NewsArticleFeatures(**sample_features)
        assert article.n_tokens_title == 10.0
        assert article.n_tokens_content == 500.0
        assert article.data_channel_is_bus == 1.0

    def test_missing_feature(self, sample_features):
        """Test that missing features raise validation error."""
        features = sample_features.copy()
        del features["n_tokens_title"]

        with pytest.raises(ValidationError) as exc_info:
            schemas.NewsArticleFeatures(**features)

        assert "n_tokens_title" in str(exc_info.value)

    def test_invalid_type(self, sample_features):
        """Test that invalid types raise validation error."""
        features = sample_features.copy()
        features["n_tokens_title"] = "invalid"  # Should be float

        with pytest.raises(ValidationError) as exc_info:
            schemas.NewsArticleFeatures(**features)

        assert "n_tokens_title" in str(exc_info.value)

    def test_extra_fields_ignored(self, sample_features):
        """Test that extra fields are ignored (default Pydantic behavior)."""
        features = sample_features.copy()
        features["extra_field"] = "should be ignored"

        # Should not raise error - Pydantic ignores extra fields by default
        article = schemas.NewsArticleFeatures(**features)
        assert not hasattr(article, "extra_field")

    def test_schema_serialization(self, sample_features):
        """Test that schema can be serialized to dict."""
        article = schemas.NewsArticleFeatures(**sample_features)
        data = article.dict()

        assert isinstance(data, dict)
        assert len(data) == 59  # All 59 features
        assert data["n_tokens_title"] == 10.0


class TestPredictionResponse:
    """Tests for PredictionResponse schema."""

    def test_valid_response(self):
        """Test that valid response is accepted."""
        response = schemas.PredictionResponse(
            predicted_shares=2500,
            log_prediction=7.824
        )

        assert response.predicted_shares == 2500
        assert response.log_prediction == 7.824

    def test_missing_field(self):
        """Test that missing field raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            schemas.PredictionResponse(predicted_shares=2500)

        assert "log_prediction" in str(exc_info.value)


class TestBatchPredictionRequest:
    """Tests for BatchPredictionRequest schema."""

    def test_valid_batch_request(self, sample_features):
        """Test that valid batch request is accepted."""
        request = schemas.BatchPredictionRequest(
            instances=[
                schemas.NewsArticleFeatures(**sample_features),
                schemas.NewsArticleFeatures(**sample_features)
            ]
        )

        assert len(request.instances) == 2

    def test_empty_batch_rejected(self):
        """Test that empty batch is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            schemas.BatchPredictionRequest(instances=[])

        assert "instances" in str(exc_info.value)

    def test_batch_size_limit(self, sample_features):
        """Test that batch size over 1000 is rejected."""
        # Create 1001 instances
        instances = [schemas.NewsArticleFeatures(**sample_features) for _ in range(1001)]

        with pytest.raises(ValidationError) as exc_info:
            schemas.BatchPredictionRequest(instances=instances)

        assert "1000" in str(exc_info.value)


class TestBatchPredictionResponse:
    """Tests for BatchPredictionResponse schema."""

    def test_valid_batch_response(self):
        """Test that valid batch response is accepted."""
        response = schemas.BatchPredictionResponse(
            predictions=[
                schemas.PredictionResponse(predicted_shares=2500, log_prediction=7.824),
                schemas.PredictionResponse(predicted_shares=1800, log_prediction=7.495),
            ],
            count=2
        )

        assert response.count == 2
        assert len(response.predictions) == 2


class TestModelInfo:
    """Tests for ModelInfo schema."""

    def test_valid_model_info(self):
        """Test that valid model info is accepted."""
        info = schemas.ModelInfo(
            status="ready",
            model_info={"model_name": "RandomForest", "load_strategy": "local"},
            features={"count": 59, "names": ["feature1", "feature2"]},
            target="shares"
        )

        assert info.status == "ready"
        assert info.model_info["model_name"] == "RandomForest"
        assert info.features["count"] == 59
        assert info.target == "shares"


class TestHealthResponse:
    """Tests for HealthResponse schema."""

    def test_healthy_status(self):
        """Test healthy status response."""
        response = schemas.HealthResponse(
            status="healthy",
            model_loaded=True,
            model_name="RandomForest",
            version="1.0.0"
        )

        assert response.status == "healthy"
        assert response.model_loaded is True
        assert response.model_name == "RandomForest"

    def test_unhealthy_status(self):
        """Test unhealthy status response."""
        response = schemas.HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_name=None,
            version="1.0.0"
        )

        assert response.status == "unhealthy"
        assert response.model_loaded is False
        assert response.model_name is None


class TestErrorResponse:
    """Tests for ErrorResponse schema."""

    def test_error_response(self):
        """Test error response schema."""
        error = schemas.ErrorResponse(
            detail="An error occurred",
            error_type="ValidationError"
        )

        assert error.detail == "An error occurred"
        assert error.error_type == "ValidationError"

    def test_error_response_optional_type(self):
        """Test error response without error_type."""
        error = schemas.ErrorResponse(detail="An error occurred")

        assert error.detail == "An error occurred"
        assert error.error_type is None
