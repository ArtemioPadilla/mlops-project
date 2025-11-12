# Getting Started with Model Serving

This guide will help you set up and run the model serving API locally and test all endpoints.

## Prerequisites

- Python 3.10
- Trained model (already included in `models/` directory)
- Dependencies installed (`pip install -r requirements.txt`)

## Installation

### 1. Install Package in Development Mode

This is **required** for the imports to work correctly:

```bash
pip install -e .
```

### 2. Install Dependencies

```bash
make requirements
```

Or directly:

```bash
pip install -r requirements.txt
```

### 3. Configure Environment (Optional)

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` to configure the model:

```bash
# Model Configuration
MODEL_NAME=RandomForestBase
MODEL_LOAD_STRATEGY=local
MODEL_PATH=models/randomforestbase_best_20251102_165526.pkl

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

## Running the Server

### Development Mode (with auto-reload)

```bash
make serve
```

Or directly:

```bash
uvicorn mlops_online_news_popularity.serving.app:app --reload --host 0.0.0.0 --port 8000
```

The server will start on `http://localhost:8000`.

### Production Mode

```bash
make serve-prod
```

This runs with 4 workers for better performance.

## Accessing the API

### Interactive Documentation

Once the server is running, open:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Health Check

Verify the server is running:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "RandomForestBase",
  "version": "1.0.0"
}
```

## Making Your First Prediction

### Using cURL

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/sample_input.json
```

Expected response:

```json
{
  "predicted_shares": 2500,
  "log_prediction": 7.824
}
```

### Using Python

```python
import requests

# Read sample data
with open("examples/sample_input.json") as f:
    features = json.load(f)

# Make prediction
response = requests.post(
    "http://localhost:8000/predict",
    json=features
)

result = response.json()
print(f"Predicted shares: {result['predicted_shares']:,}")
```

### Using the Test Scripts

We provide ready-to-use test scripts:

```bash
# Test single prediction
make test-api

# Test batch prediction
make test-api-batch

# Test CSV upload
make test-api-csv
```

## Testing All Endpoints

### 1. Root Endpoint

```bash
curl http://localhost:8000/
```

### 2. Model Information

```bash
curl http://localhost:8000/info
```

### 3. Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/sample_input.json
```

### 4. Batch Prediction (JSON)

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d @examples/sample_batch.json
```

### 5. Batch Prediction (CSV)

```bash
curl -X POST http://localhost:8000/predict/batch/csv \
  -F "file=@examples/sample_data.csv"
```

## Using Different Models

### Load from Local Pickle File

Set in `.env`:

```bash
MODEL_LOAD_STRATEGY=local
MODEL_PATH=models/kneighbors_best_20251101_235504.pkl
```

Restart the server to load the new model.

### Load from MLflow

First, find the run_id from MLflow UI:

```bash
make mlflow-ui
```

Then set in `.env`:

```bash
MODEL_LOAD_STRATEGY=mlflow
MLFLOW_RUN_ID=your-run-id-here
MLFLOW_TRACKING_URI=sqlite:///mlflow_artifacts/dev/mlflow.db
```

Restart the server.

## Troubleshooting

### Port Already in Use

If port 8000 is already in use:

```bash
# Change port in .env
API_PORT=8001

# Or specify directly
uvicorn mlops_online_news_popularity.serving.app:app --port 8001
```

### Model Not Found

If you see "Model file not found" error:

1. Check that the model file exists:
   ```bash
   ls -lh models/
   ```

2. Verify the path in `.env` or use absolute path:
   ```bash
   MODEL_PATH=/full/path/to/models/randomforestbase_best_20251102_165526.pkl
   ```

### Import Errors

If you see `ModuleNotFoundError`:

```bash
# Install package in development mode
pip install -e .
```

### Validation Errors

If predictions fail with validation errors:

1. Check that all 59 features are present
2. Verify feature names match exactly (case-sensitive)
3. Ensure values are numeric (float or int)

See the [API Reference](api-reference.md) for complete feature list.

## Next Steps

- [API Reference](api-reference.md) - Detailed endpoint documentation
- [Deployment](deployment.md) - Deploy with Docker or cloud platforms
- [Testing](testing.md) - Run automated tests
- [Troubleshooting](troubleshooting.md) - Common issues and solutions

## Quick Reference

### Common Commands

```bash
# Start server (development)
make serve

# Start server (production)
make serve-prod

# Test single prediction
make test-api

# Test batch prediction
make test-api-batch

# Test CSV upload
make test-api-csv

# View API documentation
open http://localhost:8000/docs

# Check health
curl http://localhost:8000/health
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | RandomForestBase | Model identifier |
| `MODEL_LOAD_STRATEGY` | local | "local" or "mlflow" |
| `MODEL_PATH` | models/randomforestbase_best_20251102_165526.pkl | Path to model file |
| `MLFLOW_RUN_ID` | - | MLflow run ID (if using mlflow strategy) |
| `API_HOST` | 0.0.0.0 | Server host |
| `API_PORT` | 8000 | Server port |
| `LOG_LEVEL` | INFO | Log level (DEBUG, INFO, WARNING, ERROR) |
