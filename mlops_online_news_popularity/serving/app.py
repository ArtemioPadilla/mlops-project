"""
FastAPI application for serving Online News Popularity predictions.

Provides REST API endpoints for single and batch predictions.
"""

import io
import traceback
from typing import List

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from mlops_online_news_popularity.serving import config, schemas
from mlops_online_news_popularity.serving.model_handler import get_model_handler

# API version
API_VERSION = "1.0.0"

# Create FastAPI app
app = FastAPI(
    title="Online News Popularity Prediction API",
    description="REST API for predicting social media shares of online news articles",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get model handler singleton
model_handler = get_model_handler()


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    try:
        logger.info("Initializing model handler...")
        model_handler.initialize()
        logger.info("Model handler initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        logger.error(traceback.format_exc())
        # Don't raise - allow app to start but endpoints will return errors


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API server...")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Online News Popularity Prediction API",
        "version": API_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=schemas.HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns the service status and whether the model is loaded.
    """
    try:
        is_healthy = model_handler.initialized
        return schemas.HealthResponse(
            status="healthy" if is_healthy else "unhealthy",
            model_loaded=model_handler.initialized,
            model_name=model_handler.model_name if model_handler.initialized else None,
            version=API_VERSION,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return schemas.HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_name=None,
            version=API_VERSION,
        )


@app.get("/info", response_model=schemas.ModelInfo, tags=["Info"])
async def model_info():
    """
    Get model information and metadata.

    Returns details about the loaded model including:
    - Model name and version
    - Load strategy (local/mlflow)
    - Feature information
    - Performance metrics (if available)
    """
    try:
        if not model_handler.initialized:
            raise HTTPException(
                status_code=503,
                detail="Model not initialized. Service is starting up or failed to load model.",
            )

        info = model_handler.get_model_info()
        return schemas.ModelInfo(**info)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@app.post(
    "/predict",
    response_model=schemas.PredictionResponse,
    tags=["Prediction"],
    summary="Single prediction",
)
async def predict_single(features: schemas.NewsArticleFeatures):
    """
    Make a prediction for a single news article.

    Accepts article features and returns predicted number of shares.

    Example request:
    ```json
    {
        "n_tokens_title": 10.0,
        "n_tokens_content": 500.0,
        ...
    }
    ```

    Returns:
    ```json
    {
        "predicted_shares": 2500,
        "log_prediction": 7.824
    }
    ```
    """
    try:
        if not model_handler.initialized:
            raise HTTPException(
                status_code=503,
                detail="Model not initialized. Service is starting up or failed to load model.",
            )

        # Convert to dict and make prediction
        input_data = features.dict()
        results = model_handler.handle(input_data)

        return schemas.PredictionResponse(**results[0])

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post(
    "/predict/batch",
    response_model=schemas.BatchPredictionResponse,
    tags=["Prediction"],
    summary="Batch prediction (JSON)",
)
async def predict_batch(request: schemas.BatchPredictionRequest):
    """
    Make predictions for multiple news articles (JSON format).

    Accepts a list of article features and returns predictions for all.
    Maximum batch size: 1000 instances.

    Example request:
    ```json
    {
        "instances": [
            {
                "n_tokens_title": 10.0,
                "n_tokens_content": 500.0,
                ...
            },
            {
                "n_tokens_title": 8.0,
                "n_tokens_content": 300.0,
                ...
            }
        ]
    }
    ```

    Returns:
    ```json
    {
        "predictions": [
            {"predicted_shares": 2500, "log_prediction": 7.824},
            {"predicted_shares": 1800, "log_prediction": 7.495}
        ],
        "count": 2
    }
    ```
    """
    try:
        if not model_handler.initialized:
            raise HTTPException(
                status_code=503,
                detail="Model not initialized. Service is starting up or failed to load model.",
            )

        # Convert to list of dicts
        input_data = [instance.dict() for instance in request.instances]

        # Make predictions
        results = model_handler.handle(input_data)

        return schemas.BatchPredictionResponse(
            predictions=[schemas.PredictionResponse(**r) for r in results],
            count=len(results),
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post(
    "/predict/batch/csv",
    response_model=schemas.BatchPredictionResponse,
    tags=["Prediction"],
    summary="Batch prediction (CSV upload)",
)
async def predict_batch_csv(file: UploadFile = File(...)):
    """
    Make predictions for multiple news articles from CSV file.

    Upload a CSV file with columns matching the required features.
    Returns predictions for all rows in the CSV.

    CSV must contain all 59 required features as columns.
    Maximum file size: 10MB (configurable).

    Example CSV:
    ```
    n_tokens_title,n_tokens_content,n_unique_tokens,...
    10.0,500.0,0.5,...
    8.0,300.0,0.6,...
    ```

    Returns same format as /predict/batch endpoint.
    """
    try:
        if not model_handler.initialized:
            raise HTTPException(
                status_code=503,
                detail="Model not initialized. Service is starting up or failed to load model.",
            )

        # Validate file type
        if not file.filename.endswith(".csv"):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only CSV files are accepted.",
            )

        # Read CSV file
        contents = await file.read()

        # Check file size (10MB limit)
        max_size = 10 * 1024 * 1024  # 10MB
        if len(contents) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {max_size / (1024*1024):.0f}MB",
            )

        # Parse CSV
        try:
            df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to parse CSV file: {str(e)}",
            )

        # Validate batch size
        if len(df) > 1000:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size too large. Maximum is 1000 rows, got {len(df)}",
            )

        if len(df) == 0:
            raise HTTPException(
                status_code=400,
                detail="CSV file is empty",
            )

        logger.info(f"Processing CSV with {len(df)} rows")

        # Make predictions
        results = model_handler.handle(df)

        return schemas.BatchPredictionResponse(
            predictions=[schemas.PredictionResponse(**r) for r in results],
            count=len(results),
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"CSV batch prediction failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"CSV batch prediction failed: {str(e)}"
        )


# Custom exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for uncaught exceptions."""
    logger.error(f"Uncaught exception: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred",
            "error_type": type(exc).__name__,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.API_RELOAD,
        log_level=config.LOG_LEVEL.lower(),
    )
