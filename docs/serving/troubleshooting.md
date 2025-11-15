# Troubleshooting Guide

Common issues and solutions when working with the model serving API.

## Installation Issues

### ModuleNotFoundError

**Error**:
```
ModuleNotFoundError: No module named 'mlops_online_news_popularity'
```

**Cause**: Package not installed in development mode

**Solution**:
```bash
pip install -e .
```

### Import Error for FastAPI

**Error**:
```
ModuleNotFoundError: No module named 'fastapi'
```

**Cause**: Missing dependencies

**Solution**:
```bash
make requirements
# or
pip install -r requirements.txt
```

## Server Startup Issues

### Port Already in Use

**Error**:
```
ERROR: [Errno 48] error while attempting to bind on address ('0.0.0.0', 8000): address already in use
```

**Cause**: Another process is using port 8000

**Solutions**:

1. **Find and kill the process**:
   ```bash
   # Find process using port 8000
   lsof -i :8000
   # Kill it
   kill -9 <PID>
   ```

2. **Use a different port**:
   ```bash
   # Set in .env
   API_PORT=8001

   # Or specify directly
   uvicorn mlops_online_news_popularity.serving.app:app --port 8001
   ```

### Model File Not Found

**Error**:
```
FileNotFoundError: Model file not found: models/randomforestbase_best_20251102_165526.pkl
```

**Cause**: Model file doesn't exist or path is incorrect

**Solutions**:

1. **Check if file exists**:
   ```bash
   ls -lh models/
   ```

2. **Use absolute path**:
   ```bash
   MODEL_PATH=/full/path/to/mlops-project/models/randomforestbase_best_20251102_165526.pkl
   ```

3. **Train a new model**:
   ```bash
   make train
   ```

### MLflow Model Loading Fails

**Error**:
```
ValueError: MLFLOW_RUN_ID must be provided when using mlflow load strategy
```

**Cause**: Missing MLflow run ID

**Solutions**:

1. **Find run ID from MLflow UI**:
   ```bash
   make mlflow-ui
   # Open http://localhost:5001
   # Click on a run and copy the run ID
   ```

2. **Set in .env**:
   ```bash
   MODEL_LOAD_STRATEGY=mlflow
   MLFLOW_RUN_ID=abc123def456
   ```

## Request/Response Issues

### 422 Validation Error

**Error Response**:
```json
{
  "detail": [
    {
      "loc": ["body", "n_tokens_title"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**Cause**: Missing or invalid input fields

**Solutions**:

1. **Check all 59 features are present**:
   ```python
   from mlops_online_news_popularity.serving import config
   print(f"Required features: {len(config.FEATURE_NAMES)}")
   print(config.FEATURE_NAMES)
   ```

2. **Use example files**:
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d @examples/sample_input.json
   ```

3. **Validate input structure**:
   ```python
   import json
   from mlops_online_news_popularity.serving.schemas import NewsArticleFeatures

   with open("your_input.json") as f:
       data = json.load(f)

   # This will raise validation error if invalid
   features = NewsArticleFeatures(**data)
   ```

### 400 Bad Request - CSV Upload

**Error**:
```json
{
  "detail": "Invalid file type. Only CSV files are accepted."
}
```

**Cause**: File is not a CSV or has wrong extension

**Solution**:
```bash
# Ensure file has .csv extension
mv data.txt data.csv

# Upload
curl -X POST http://localhost:8000/predict/batch/csv \
  -F "file=@data.csv"
```

### 400 Bad Request - Missing Features in CSV

**Error**:
```json
{
  "detail": "Missing required features: ['n_tokens_title', 'n_tokens_content']"
}
```

**Cause**: CSV doesn't have all 59 required columns

**Solution**:

1. **Check CSV headers**:
   ```bash
   head -1 examples/sample_data.csv
   ```

2. **Ensure all 59 features are present** as columns

3. **Use correct feature names** (case-sensitive)

### 500 Internal Server Error

**Error**:
```json
{
  "detail": "Prediction failed: ..."
}
```

**Cause**: Model prediction failed

**Solutions**:

1. **Check server logs**:
   ```bash
   # If running locally
   # Logs appear in terminal

   # If Docker
   docker logs online-news-predictor

   # If docker-compose
   docker-compose logs -f
   ```

2. **Verify model is loaded**:
   ```bash
   curl http://localhost:8000/info
   ```

3. **Test with known good data**:
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d @examples/sample_input.json
   ```

### 503 Service Unavailable

**Error**:
```json
{
  "detail": "Model not initialized. Service is starting up or failed to load model."
}
```

**Cause**: Model hasn't been loaded yet or failed to load

**Solutions**:

1. **Wait for startup** (can take 2-5 seconds for large models)

2. **Check health endpoint**:
   ```bash
   curl http://localhost:8000/health
   ```

3. **Verify model file exists and is readable**

4. **Check startup logs** for errors

## Docker Issues

### Container Exits Immediately

**Symptom**: Container starts but immediately exits

**Diagnosis**:
```bash
docker logs online-news-predictor
```

**Common Causes & Solutions**:

1. **Model file not found**:
   ```bash
   # Ensure volume mount is correct
   docker run -p 8000:8000 \
     -v $(pwd)/models:/app/models:ro \
     ml-service:latest
   ```

2. **Port conflict**:
   ```bash
   # Use different port
   docker run -p 8001:8000 ml-service:latest
   ```

3. **Missing environment variables**:
   ```bash
   docker run -p 8000:8000 \
     -e MODEL_NAME=RandomForestBase \
     -e MODEL_PATH=/app/models/randomforestbase_best_20251102_165526.pkl \
     -v $(pwd)/models:/app/models:ro \
     ml-service:latest
   ```

### Cannot Access API from Host

**Symptom**: `curl: (7) Failed to connect to localhost port 8000: Connection refused`

**Solutions**:

1. **Check container is running**:
   ```bash
   docker ps
   ```

2. **Verify port mapping**:
   ```bash
   docker ps -a
   # Look for: 0.0.0.0:8000->8000/tcp
   ```

3. **Check firewall settings**

4. **Try 127.0.0.1 instead of localhost**:
   ```bash
   curl http://127.0.0.1:8000/health
   ```

### High Memory Usage

**Symptom**: Container uses > 2GB RAM

**Cause**: Large model (RandomForest is 234MB) + multiple requests

**Solutions**:

1. **Use a smaller model**:
   ```bash
   # Ridge: 11KB
   MODEL_PATH=/app/models/ridge_20251101_222219.pkl

   # KNeighbors: 13MB
   MODEL_PATH=/app/models/kneighbors_best_20251101_235504.pkl
   ```

2. **Limit container memory**:
   ```bash
   docker run -p 8000:8000 \
     --memory="1g" \
     -v $(pwd)/models:/app/models:ro \
     ml-service:latest
   ```

3. **Reduce workers** (if using gunicorn):
   ```bash
   # In Dockerfile, reduce --workers from 4 to 2
   ```

### Volume Mount Not Working

**Symptom**: Model file not found despite volume mount

**Solutions**:

1. **Use absolute paths**:
   ```bash
   docker run -p 8000:8000 \
     -v /full/path/to/models:/app/models:ro \
     ml-service:latest
   ```

2. **Check file permissions**:
   ```bash
   ls -l models/
   # Should be readable by all users
   ```

3. **Verify path inside container**:
   ```bash
   docker exec online-news-predictor ls -l /app/models
   ```

## Performance Issues

### Slow Predictions

**Symptom**: Predictions take > 1 second

**Causes & Solutions**:

1. **Large model (RandomForest: 234MB)**:
   - Use smaller model (Ridge, KNeighbors)
   - Ensure model is cached in memory

2. **Too many features being processed**:
   - This is expected (59 features)
   - Use batch endpoint for multiple predictions

3. **Not enough resources**:
   - Increase CPU/memory allocation
   - Scale horizontally (multiple containers)

### High Latency

**Symptom**: Response time > 500ms

**Solutions**:

1. **Use batch endpoints** for multiple predictions

2. **Enable HTTP/2**:
   ```python
   # In production, use a reverse proxy with HTTP/2
   ```

3. **Add caching** (if predictions are repeated):
   ```python
   from functools import lru_cache

   @lru_cache(maxsize=1000)
   def cached_predict(features_hash):
       return model.predict(features)
   ```

4. **Use load balancer** with multiple instances

## Testing Issues

### Tests Fail with "Model Not Found"

**Error**:
```
FileNotFoundError: Model file not found
```

**Cause**: Tests use mock models, not real ones

**Solution**: This is expected. Tests use fixtures from `conftest.py` that create temporary mock models.

### Tests Pass Locally But Fail in CI

**Causes & Solutions**:

1. **Missing dependencies**:
   ```yaml
   # In CI config, ensure all dependencies installed
   pip install -e .
   pip install -r requirements.txt
   ```

2. **Different Python version**:
   ```yaml
   # Specify Python 3.10
   python-version: '3.10'
   ```

3. **File paths**:
   - Use relative paths in tests
   - Don't assume specific directory structure

### Coverage Not Generated

**Solution**:
```bash
# Ensure pytest-cov is installed
pip install pytest-cov

# Run with coverage
pytest --cov=mlops_online_news_popularity --cov-report=html
```

## Configuration Issues

### Environment Variables Not Loaded

**Symptom**: API uses default values instead of `.env` values

**Solutions**:

1. **Ensure .env file exists**:
   ```bash
   ls -la .env
   ```

2. **Restart server** after changing .env

3. **Check .env format**:
   ```bash
   # Correct
   MODEL_NAME=RandomForestBase

   # Incorrect (no spaces)
   MODEL_NAME = RandomForestBase
   ```

4. **Use python-dotenv**:
   ```python
   from dotenv import load_dotenv
   load_dotenv()  # This is already in config.py
   ```

### Can't Switch Models

**Symptom**: API still uses old model after changing configuration

**Solution**:

1. **Restart the server**:
   ```bash
   # Local
   # Stop with Ctrl+C, then restart with make serve

   # Docker
   docker restart online-news-predictor

   # docker-compose
   docker-compose restart
   ```

2. **Clear Python cache**:
   ```bash
   find . -type d -name "__pycache__" -exec rm -rf {} +
   find . -type f -name "*.pyc" -delete
   ```

## Getting Help

If you're still stuck:

1. **Check server logs** for detailed error messages

2. **Enable debug logging**:
   ```bash
   LOG_LEVEL=DEBUG make serve
   ```

3. **Run tests** to isolate the issue:
   ```bash
   make test-serving
   ```

4. **Consult API documentation**:
   - Swagger UI: `http://localhost:8000/docs`
   - This documentation: [API Reference](api-reference.md)

5. **Check GitHub Issues**: [mlops-project/issues](https://github.com/ArtemioPadilla/mlops-project/issues)

6. **Enable verbose output**:
   ```bash
   uvicorn mlops_online_news_popularity.serving.app:app --log-level debug
   ```

## Common Error Codes

| Status Code | Meaning | Common Causes |
|-------------|---------|---------------|
| 400 | Bad Request | Invalid input, wrong file type |
| 404 | Not Found | Wrong endpoint URL |
| 405 | Method Not Allowed | Wrong HTTP method (GET vs POST) |
| 422 | Validation Error | Missing features, wrong types |
| 500 | Internal Server Error | Model prediction failed, server bug |
| 503 | Service Unavailable | Model not loaded yet |

## Quick Diagnostic Checklist

When something goes wrong, check:

- [ ] Server is running (`curl http://localhost:8000/health`)
- [ ] Model is loaded (`curl http://localhost:8000/info`)
- [ ] All 59 features are present in request
- [ ] Feature names match exactly (case-sensitive)
- [ ] Values are numeric (float/int)
- [ ] Using correct HTTP method (POST for predictions)
- [ ] Content-Type header is set
- [ ] Check server logs for errors
- [ ] Restart server if configuration changed

## Still Need Help?

Open an issue with:
- Error message (full stack trace)
- Server logs
- Request you're trying to make
- Expected vs actual behavior
- Python version, OS, Docker version (if applicable)
