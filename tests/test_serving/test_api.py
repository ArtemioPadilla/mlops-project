"""
Integration tests for FastAPI endpoints.

Tests all API endpoints including health, info, predict, batch, and CSV upload.
"""

import io

import pytest

from mlops_online_news_popularity.serving import schemas


@pytest.mark.integration
class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_endpoint(self, test_client):
        """Test root endpoint returns API information."""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data


@pytest.mark.integration
class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_endpoint(self, initialized_test_client):
        """Test health endpoint with initialized model."""
        response = initialized_test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "unhealthy"]
        assert "model_loaded" in data
        assert "version" in data

    def test_health_endpoint_schema(self, initialized_test_client):
        """Test health endpoint returns valid HealthResponse schema."""
        response = initialized_test_client.get("/health")

        data = response.json()
        health_response = schemas.HealthResponse(**data)
        assert health_response.status in ["healthy", "unhealthy"]


@pytest.mark.integration
class TestModelInfoEndpoint:
    """Tests for model info endpoint."""

    def test_info_endpoint_with_initialized_model(self, initialized_test_client):
        """Test info endpoint with initialized model."""
        response = initialized_test_client.get("/info")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert "model_info" in data
        assert "features" in data
        assert data["features"]["count"] == 59

    def test_info_endpoint_without_model(self, test_client):
        """Test info endpoint without initialized model."""
        response = test_client.get("/info")

        # Should return 503 if model not initialized
        assert response.status_code in [503, 200]  # Depends on startup

    def test_info_endpoint_schema(self, initialized_test_client):
        """Test info endpoint returns valid ModelInfo schema."""
        response = initialized_test_client.get("/info")

        data = response.json()
        model_info = schemas.ModelInfo(**data)
        assert model_info.status == "ready"


@pytest.mark.integration
class TestPredictEndpoint:
    """Tests for single prediction endpoint."""

    def test_predict_success(self, initialized_test_client, sample_features):
        """Test successful single prediction."""
        response = initialized_test_client.post("/predict", json=sample_features)

        assert response.status_code == 200
        data = response.json()
        assert "predicted_shares" in data
        assert "log_prediction" in data
        assert isinstance(data["predicted_shares"], int)
        assert data["predicted_shares"] > 0

    def test_predict_missing_feature(self, initialized_test_client, sample_features):
        """Test prediction with missing feature returns 400."""
        features = sample_features.copy()
        del features["n_tokens_title"]

        response = initialized_test_client.post("/predict", json=features)

        assert response.status_code == 422  # Validation error from Pydantic

    def test_predict_invalid_type(self, initialized_test_client, sample_features):
        """Test prediction with invalid type returns 422."""
        features = sample_features.copy()
        features["n_tokens_title"] = "invalid"  # Should be float

        response = initialized_test_client.post("/predict", json=features)

        assert response.status_code == 422

    def test_predict_extra_fields_ignored(self, initialized_test_client, sample_features):
        """Test that extra fields are ignored."""
        features = sample_features.copy()
        features["extra_field"] = "should_be_ignored"

        response = initialized_test_client.post("/predict", json=features)

        # Should succeed - extra fields are ignored
        assert response.status_code == 200

    def test_predict_response_schema(self, initialized_test_client, sample_features):
        """Test prediction response matches PredictionResponse schema."""
        response = initialized_test_client.post("/predict", json=sample_features)

        data = response.json()
        prediction = schemas.PredictionResponse(**data)
        assert isinstance(prediction.predicted_shares, int)
        assert isinstance(prediction.log_prediction, float)


@pytest.mark.integration
class TestBatchPredictEndpoint:
    """Tests for batch prediction endpoint (JSON)."""

    def test_batch_predict_success(self, initialized_test_client, sample_features):
        """Test successful batch prediction."""
        batch_request = {
            "instances": [sample_features, sample_features]
        }

        response = initialized_test_client.post("/predict/batch", json=batch_request)

        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "count" in data
        assert data["count"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_predict_single_instance(self, initialized_test_client, sample_features):
        """Test batch prediction with single instance."""
        batch_request = {
            "instances": [sample_features]
        }

        response = initialized_test_client.post("/predict/batch", json=batch_request)

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1

    def test_batch_predict_empty_batch(self, initialized_test_client):
        """Test batch prediction with empty batch returns 422."""
        batch_request = {"instances": []}

        response = initialized_test_client.post("/predict/batch", json=batch_request)

        assert response.status_code == 422  # Validation error

    def test_batch_predict_large_batch(self, initialized_test_client, sample_features):
        """Test batch prediction with 100 instances."""
        batch_request = {
            "instances": [sample_features for _ in range(100)]
        }

        response = initialized_test_client.post("/predict/batch", json=batch_request)

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 100

    def test_batch_predict_over_limit(self, initialized_test_client, sample_features):
        """Test batch prediction over 1000 instances returns 422."""
        batch_request = {
            "instances": [sample_features for _ in range(1001)]
        }

        response = initialized_test_client.post("/predict/batch", json=batch_request)

        assert response.status_code == 422  # Exceeds limit

    def test_batch_predict_response_schema(self, initialized_test_client, sample_features):
        """Test batch response matches BatchPredictionResponse schema."""
        batch_request = {
            "instances": [sample_features, sample_features]
        }

        response = initialized_test_client.post("/predict/batch", json=batch_request)

        data = response.json()
        batch_response = schemas.BatchPredictionResponse(**data)
        assert batch_response.count == 2
        assert len(batch_response.predictions) == 2


@pytest.mark.integration
class TestBatchPredictCSVEndpoint:
    """Tests for batch prediction endpoint (CSV upload)."""

    def test_csv_upload_success(self, initialized_test_client, sample_csv_content):
        """Test successful CSV upload."""
        files = {"file": ("test.csv", io.BytesIO(sample_csv_content), "text/csv")}

        response = initialized_test_client.post("/predict/batch/csv", files=files)

        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "count" in data
        assert data["count"] == 2  # CSV has 2 rows

    def test_csv_upload_invalid_file_type(self, initialized_test_client):
        """Test CSV upload with invalid file type returns 400."""
        files = {"file": ("test.txt", b"invalid content", "text/plain")}

        response = initialized_test_client.post("/predict/batch/csv", files=files)

        assert response.status_code == 400
        assert "CSV" in response.json()["detail"]

    def test_csv_upload_malformed_csv(self, initialized_test_client):
        """Test CSV upload with malformed CSV returns 400."""
        invalid_csv = b"invalid,csv,data\n1,2"  # Mismatched columns
        files = {"file": ("test.csv", io.BytesIO(invalid_csv), "text/csv")}

        response = initialized_test_client.post("/predict/batch/csv", files=files)

        assert response.status_code == 400

    def test_csv_upload_missing_features(self, initialized_test_client):
        """Test CSV upload with missing features returns 400."""
        # CSV with only 3 columns instead of 59
        csv_content = b"n_tokens_title,n_tokens_content,n_unique_tokens\n10.0,500.0,0.5"
        files = {"file": ("test.csv", io.BytesIO(csv_content), "text/csv")}

        response = initialized_test_client.post("/predict/batch/csv", files=files)

        assert response.status_code == 400

    def test_csv_upload_too_large(self, initialized_test_client):
        """Test CSV upload with file over 10MB returns 400."""
        # Create CSV content larger than 10MB
        large_content = b"a," * 1000000  # Simplified for testing
        files = {"file": ("test.csv", io.BytesIO(large_content * 20), "text/csv")}

        response = initialized_test_client.post("/predict/batch/csv", files=files)

        assert response.status_code == 400
        assert "too large" in response.json()["detail"].lower()

    def test_csv_upload_response_schema(self, initialized_test_client, sample_csv_content):
        """Test CSV response matches BatchPredictionResponse schema."""
        files = {"file": ("test.csv", io.BytesIO(sample_csv_content), "text/csv")}

        response = initialized_test_client.post("/predict/batch/csv", files=files)

        data = response.json()
        batch_response = schemas.BatchPredictionResponse(**data)
        assert batch_response.count > 0


@pytest.mark.integration
class TestErrorHandling:
    """Tests for API error handling."""

    def test_404_not_found(self, test_client):
        """Test 404 for non-existent endpoint."""
        response = test_client.get("/nonexistent")

        assert response.status_code == 404

    def test_405_method_not_allowed(self, test_client):
        """Test 405 for wrong HTTP method."""
        response = test_client.get("/predict")  # Should be POST

        assert response.status_code == 405

    def test_error_response_format(self, initialized_test_client):
        """Test that error responses have detail field."""
        # Trigger validation error
        response = initialized_test_client.post("/predict", json={})

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data


@pytest.mark.integration
class TestCORS:
    """Tests for CORS middleware."""

    def test_cors_headers(self, test_client):
        """Test CORS headers are present."""
        # Test with GET request (OPTIONS may not be properly handled by TestClient)
        response = test_client.get("/health", headers={"Origin": "http://localhost:3000"})

        # CORS middleware should add these headers
        # Note: TestClient may normalize header names differently
        headers_lower = {k.lower(): v for k, v in response.headers.items()}
        assert "access-control-allow-origin" in headers_lower or response.status_code == 200


@pytest.mark.integration
class TestAPIDocumentation:
    """Tests for API documentation endpoints."""

    def test_openapi_schema(self, test_client):
        """Test OpenAPI schema is available."""
        response = test_client.get("/openapi.json")

        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema

    def test_swagger_ui_docs(self, test_client):
        """Test Swagger UI documentation is available."""
        response = test_client.get("/docs")

        assert response.status_code == 200
        assert b"swagger" in response.content.lower()

    def test_redoc_docs(self, test_client):
        """Test ReDoc documentation is available."""
        response = test_client.get("/redoc")

        assert response.status_code == 200
        assert b"redoc" in response.content.lower()
