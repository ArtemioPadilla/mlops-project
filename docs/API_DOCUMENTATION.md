# Online News Popularity Prediction API - Documentation

## Overview

The Online News Popularity Prediction API is a production-ready REST API built with FastAPI that serves machine learning models for predicting the number of social media shares an online news article will receive.

**Base URL**: `http://localhost:8000`

**API Version**: 1.0.0

## Features

- **Single Prediction**: Predict shares for a single news article
- **Batch Prediction**: Process multiple articles in one request (JSON or CSV)
- **Health Monitoring**: Built-in health check endpoint
- **Model Information**: Query model metadata and configuration
- **Interactive Documentation**: Auto-generated Swagger UI and ReDoc
- **Input Validation**: Pydantic-based schema validation with detailed error messages
- **Logging**: Comprehensive logging for debugging and monitoring

## Authentication

Currently, the API does not require authentication. For production deployments, consider adding:
- API Keys
- OAuth2
- JWT tokens

## Endpoints

### Root Endpoint

#### `GET /`

Returns API information and links to documentation.

**Response**:
```json
{
  "name": "Online News Popularity Prediction API",
  "version": "1.0.0",
  "status": "running",
  "docs": "/docs",
  "health": "/health"
}
```

---

### Health Check

#### `GET /health`

Check if the API service is running and the model is loaded.

**Response**: `200 OK`
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "RandomForestBase",
  "version": "1.0.0"
}
```

**Example**:
```bash
curl http://localhost:8000/health
```

---

### Model Information

#### `GET /info`

Get detailed information about the loaded model, including features and configuration.

**Response**: `200 OK`
```json
{
  "status": "ready",
  "model_info": {
    "model_name": "RandomForestBase",
    "load_strategy": "local",
    "model_size_mb": 234.5,
    "model_path": "/app/models/randomforestbase_best_20251102_165526.pkl"
  },
  "features": {
    "count": 59,
    "names": [
      "n_tokens_title",
      "n_tokens_content",
      "..."
    ]
  },
  "target": "shares"
}
```

**Example**:
```bash
curl http://localhost:8000/info
```

---

### Single Prediction

#### `POST /predict`

Make a prediction for a single news article.

**Request Body**: JSON object with 59 features

**Content-Type**: `application/json`

**Request Schema**:
```json
{
  "n_tokens_title": float,
  "n_tokens_content": float,
  "n_unique_tokens": float,
  "n_non_stop_words": float,
  "n_non_stop_unique_tokens": float,
  "num_hrefs": float,
  "num_self_hrefs": float,
  "num_imgs": float,
  "num_videos": float,
  "average_token_length": float,
  "num_keywords": float,
  "data_channel_is_lifestyle": float,
  "data_channel_is_entertainment": float,
  "data_channel_is_bus": float,
  "data_channel_is_socmed": float,
  "data_channel_is_tech": float,
  "data_channel_is_world": float,
  "kw_min_min": float,
  "kw_max_min": float,
  "kw_avg_min": float,
  "kw_min_max": float,
  "kw_max_max": float,
  "kw_avg_max": float,
  "kw_min_avg": float,
  "kw_max_avg": float,
  "kw_avg_avg": float,
  "self_reference_min_shares": float,
  "self_reference_max_shares": float,
  "self_reference_avg_sharess": float,
  "weekday_is_monday": float,
  "weekday_is_tuesday": float,
  "weekday_is_wednesday": float,
  "weekday_is_thursday": float,
  "weekday_is_friday": float,
  "weekday_is_saturday": float,
  "weekday_is_sunday": float,
  "is_weekend": float,
  "LDA_00": float,
  "LDA_01": float,
  "LDA_02": float,
  "LDA_03": float,
  "LDA_04": float,
  "global_subjectivity": float,
  "global_sentiment_polarity": float,
  "global_rate_positive_words": float,
  "global_rate_negative_words": float,
  "rate_positive_words": float,
  "rate_negative_words": float,
  "avg_positive_polarity": float,
  "min_positive_polarity": float,
  "max_positive_polarity": float,
  "avg_negative_polarity": float,
  "min_negative_polarity": float,
  "max_negative_polarity": float,
  "title_subjectivity": float,
  "title_sentiment_polarity": float,
  "abs_title_subjectivity": float,
  "abs_title_sentiment_polarity": float,
  "mixed_type_col": float
}
```

**Response**: `200 OK`
```json
{
  "predicted_shares": 2500,
  "log_prediction": 7.824
}
```

**Example**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/sample_input.json
```

**Python Example**:
```python
import requests

features = {
    "n_tokens_title": 10.0,
    "n_tokens_content": 500.0,
    # ... (all 59 features)
}

response = requests.post(
    "http://localhost:8000/predict",
    json=features
)

result = response.json()
print(f"Predicted shares: {result['predicted_shares']:,}")
```

**Errors**:
- `400 Bad Request`: Invalid input (missing features, wrong types)
- `500 Internal Server Error`: Model prediction failed
- `503 Service Unavailable`: Model not loaded

---

### Batch Prediction (JSON)

#### `POST /predict/batch`

Make predictions for multiple news articles at once.

**Request Body**: JSON object with array of instances

**Content-Type**: `application/json`

**Limits**: Maximum 1000 instances per request

**Request Schema**:
```json
{
  "instances": [
    {
      "n_tokens_title": 10.0,
      "n_tokens_content": 500.0,
      "..."
    },
    {
      "n_tokens_title": 8.0,
      "n_tokens_content": 300.0,
      "..."
    }
  ]
}
```

**Response**: `200 OK`
```json
{
  "predictions": [
    {
      "predicted_shares": 2500,
      "log_prediction": 7.824
    },
    {
      "predicted_shares": 1800,
      "log_prediction": 7.495
    }
  ],
  "count": 2
}
```

**Example**:
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d @examples/sample_batch.json
```

**Python Example**:
```python
import requests

batch_request = {
    "instances": [
        {"n_tokens_title": 10.0, "n_tokens_content": 800.0, ...},
        {"n_tokens_title": 8.0, "n_tokens_content": 600.0, ...}
    ]
}

response = requests.post(
    "http://localhost:8000/predict/batch",
    json=batch_request
)

result = response.json()
for i, pred in enumerate(result["predictions"], 1):
    print(f"Article {i}: {pred['predicted_shares']:,} shares")
```

**Errors**:
- `400 Bad Request`: Invalid input, batch too large (>1000), empty batch
- `500 Internal Server Error`: Prediction failed
- `503 Service Unavailable`: Model not loaded

---

### Batch Prediction (CSV Upload)

#### `POST /predict/batch/csv`

Upload a CSV file for batch predictions.

**Request Body**: Multipart form data with CSV file

**Content-Type**: `multipart/form-data`

**Limits**:
- Maximum file size: 10MB
- Maximum rows: 1000

**CSV Format**:
- Header row required
- Columns must match the 59 required features
- Values must be numeric

**Example CSV**:
```csv
n_tokens_title,n_tokens_content,n_unique_tokens,...
10.0,500.0,0.5,...
8.0,300.0,0.6,...
```

**Response**: `200 OK`
```json
{
  "predictions": [
    {
      "predicted_shares": 2500,
      "log_prediction": 7.824
    },
    {
      "predicted_shares": 1800,
      "log_prediction": 7.495
    }
  ],
  "count": 2
}
```

**Example**:
```bash
curl -X POST http://localhost:8000/predict/batch/csv \
  -F "file=@examples/sample_data.csv"
```

**Python Example**:
```python
import requests

with open("examples/sample_data.csv", "rb") as f:
    files = {"file": ("data.csv", f, "text/csv")}
    response = requests.post(
        "http://localhost:8000/predict/batch/csv",
        files=files
    )

result = response.json()
print(f"Processed {result['count']} articles")
```

**Errors**:
- `400 Bad Request`: Invalid file type, file too large, parsing error, batch too large
- `500 Internal Server Error`: Prediction failed
- `503 Service Unavailable`: Model not loaded

---

## Feature Descriptions

### Content Features

| Feature | Description | Type |
|---------|-------------|------|
| `n_tokens_title` | Number of words in the title | float |
| `n_tokens_content` | Number of words in the content | float |
| `n_unique_tokens` | Rate of unique words in the content | float |
| `n_non_stop_words` | Rate of non-stop words in the content | float |
| `n_non_stop_unique_tokens` | Rate of unique non-stop words | float |
| `num_hrefs` | Number of links | float |
| `num_self_hrefs` | Number of links to other Mashable articles | float |
| `num_imgs` | Number of images | float |
| `num_videos` | Number of videos | float |
| `average_token_length` | Average length of words | float |
| `num_keywords` | Number of keywords in metadata | float |

### Channel Features (Binary: 0 or 1)

| Feature | Description |
|---------|-------------|
| `data_channel_is_lifestyle` | Is data channel 'Lifestyle'? |
| `data_channel_is_entertainment` | Is data channel 'Entertainment'? |
| `data_channel_is_bus` | Is data channel 'Business'? |
| `data_channel_is_socmed` | Is data channel 'Social Media'? |
| `data_channel_is_tech` | Is data channel 'Tech'? |
| `data_channel_is_world` | Is data channel 'World'? |

### Keyword Features

| Feature | Description |
|---------|-------------|
| `kw_min_min` | Worst keyword (min. shares) |
| `kw_max_min` | Worst keyword (max. shares) |
| `kw_avg_min` | Worst keyword (avg. shares) |
| `kw_min_max` | Best keyword (min. shares) |
| `kw_max_max` | Best keyword (max. shares) |
| `kw_avg_max` | Best keyword (avg. shares) |
| `kw_min_avg` | Avg. keyword (min. shares) |
| `kw_max_avg` | Avg. keyword (max. shares) |
| `kw_avg_avg` | Avg. keyword (avg. shares) |

### Self-Reference Features

| Feature | Description |
|---------|-------------|
| `self_reference_min_shares` | Min. shares of referenced articles |
| `self_reference_max_shares` | Max. shares of referenced articles |
| `self_reference_avg_sharess` | Avg. shares of referenced articles |

### Temporal Features (Binary: 0 or 1)

| Feature | Description |
|---------|-------------|
| `weekday_is_monday` | Was published on Monday? |
| `weekday_is_tuesday` | Was published on Tuesday? |
| `weekday_is_wednesday` | Was published on Wednesday? |
| `weekday_is_thursday` | Was published on Thursday? |
| `weekday_is_friday` | Was published on Friday? |
| `weekday_is_saturday` | Was published on Saturday? |
| `weekday_is_sunday` | Was published on Sunday? |
| `is_weekend` | Was published on the weekend? |

### LDA Topic Features

| Feature | Description |
|---------|-------------|
| `LDA_00` | Closeness to LDA topic 0 |
| `LDA_01` | Closeness to LDA topic 1 |
| `LDA_02` | Closeness to LDA topic 2 |
| `LDA_03` | Closeness to LDA topic 3 |
| `LDA_04` | Closeness to LDA topic 4 |

### Sentiment Features

| Feature | Description |
|---------|-------------|
| `global_subjectivity` | Text subjectivity |
| `global_sentiment_polarity` | Text sentiment polarity |
| `global_rate_positive_words` | Rate of positive words |
| `global_rate_negative_words` | Rate of negative words |
| `rate_positive_words` | Rate of positive words among non-neutral |
| `rate_negative_words` | Rate of negative words among non-neutral |
| `avg_positive_polarity` | Avg. polarity of positive words |
| `min_positive_polarity` | Min. polarity of positive words |
| `max_positive_polarity` | Max. polarity of positive words |
| `avg_negative_polarity` | Avg. polarity of negative words |
| `min_negative_polarity` | Min. polarity of negative words |
| `max_negative_polarity` | Max. polarity of negative words |

### Title Features

| Feature | Description |
|---------|-------------|
| `title_subjectivity` | Title subjectivity |
| `title_sentiment_polarity` | Title sentiment polarity |
| `abs_title_subjectivity` | Absolute subjectivity level |
| `abs_title_sentiment_polarity` | Absolute polarity level |

### Other

| Feature | Description |
|---------|-------------|
| `mixed_type_col` | Mixed type column |

---

## Error Codes

| Status Code | Description |
|-------------|-------------|
| `200 OK` | Request successful |
| `400 Bad Request` | Invalid input (missing features, wrong format, validation error) |
| `404 Not Found` | Endpoint not found |
| `500 Internal Server Error` | Server error during prediction |
| `503 Service Unavailable` | Model not loaded or initialization failed |

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong",
  "error_type": "ValidationError"
}
```

---

## Rate Limiting

Currently, no rate limiting is implemented. For production:
- Consider implementing rate limiting per IP or API key
- Recommended: 100 requests per minute per client
- Batch endpoints: 10 requests per minute per client

---

## Performance Considerations

### Response Times (Approximate)

- Health check: <10ms
- Model info: <10ms
- Single prediction: 50-100ms
- Batch prediction (100 articles): 200-500ms
- CSV upload (100 articles): 300-600ms

### Optimization Tips

1. **Use batch endpoints** for multiple predictions
2. **CSV format** is slightly faster than JSON for large batches
3. **Keep batches under 500 items** for optimal performance
4. **Reuse connections** when making multiple requests

---

## Interactive Documentation

FastAPI automatically generates interactive API documentation:

### Swagger UI
**URL**: `http://localhost:8000/docs`

Features:
- Try out all endpoints directly in the browser
- See request/response schemas
- Download OpenAPI spec

### ReDoc
**URL**: `http://localhost:8000/redoc`

Features:
- Clean, organized documentation
- Schema navigation
- Code samples

---

## Configuration

The API can be configured via environment variables (`.env` file):

```bash
# Model Configuration
MODEL_NAME=RandomForestBase
MODEL_LOAD_STRATEGY=local  # local or mlflow
MODEL_PATH=models/randomforestbase_best_20251102_165526.pkl

# MLflow (if MODEL_LOAD_STRATEGY=mlflow)
MLFLOW_RUN_ID=your-run-id
MLFLOW_TRACKING_URI=sqlite:///mlflow_artifacts/dev/mlflow.db

# Server Settings
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

---

## Deployment

### Local Development
```bash
make serve
```

### Production
```bash
make serve-prod
```

### Docker
```bash
make docker-build
make docker-run
```

### Docker Compose
```bash
make docker-up
```

---

## Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/ArtemioPadilla/mlops-project/issues
- Documentation: See README.md for complete setup instructions

---

## Version History

### v1.0.0 (Current)
- Initial API release
- Single and batch prediction endpoints
- CSV upload support
- Health monitoring
- Interactive documentation
