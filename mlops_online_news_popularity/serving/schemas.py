"""
Pydantic schemas for API request/response validation.

Defines the data models for all API endpoints with proper validation and documentation.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

from mlops_online_news_popularity.serving import config


class NewsArticleFeatures(BaseModel):
    """
    Input features for a single news article prediction.

    All 59 features required by the model.
    """

    # Content features
    n_tokens_title: float = Field(..., description="Number of words in the title")
    n_tokens_content: float = Field(..., description="Number of words in the content")
    n_unique_tokens: float = Field(..., description="Rate of unique words in the content")
    n_non_stop_words: float = Field(..., description="Rate of non-stop words in the content")
    n_non_stop_unique_tokens: float = Field(
        ..., description="Rate of unique non-stop words in the content"
    )
    num_hrefs: float = Field(..., description="Number of links")
    num_self_hrefs: float = Field(..., description="Number of links to other Mashable articles")
    num_imgs: float = Field(..., description="Number of images")
    num_videos: float = Field(..., description="Number of videos")
    average_token_length: float = Field(..., description="Average length of the words in the content")
    num_keywords: float = Field(..., description="Number of keywords in the metadata")

    # Channel features (binary flags)
    data_channel_is_lifestyle: float = Field(..., description="Is data channel 'Lifestyle'?")
    data_channel_is_entertainment: float = Field(..., description="Is data channel 'Entertainment'?")
    data_channel_is_bus: float = Field(..., description="Is data channel 'Business'?")
    data_channel_is_socmed: float = Field(..., description="Is data channel 'Social Media'?")
    data_channel_is_tech: float = Field(..., description="Is data channel 'Tech'?")
    data_channel_is_world: float = Field(..., description="Is data channel 'World'?")

    # Keyword features
    kw_min_min: float = Field(..., description="Worst keyword (min. shares)")
    kw_max_min: float = Field(..., description="Worst keyword (max. shares)")
    kw_avg_min: float = Field(..., description="Worst keyword (avg. shares)")
    kw_min_max: float = Field(..., description="Best keyword (min. shares)")
    kw_max_max: float = Field(..., description="Best keyword (max. shares)")
    kw_avg_max: float = Field(..., description="Best keyword (avg. shares)")
    kw_min_avg: float = Field(..., description="Avg. keyword (min. shares)")
    kw_max_avg: float = Field(..., description="Avg. keyword (max. shares)")
    kw_avg_avg: float = Field(..., description="Avg. keyword (avg. shares)")

    # Self-reference features
    self_reference_min_shares: float = Field(..., description="Min. shares of referenced articles")
    self_reference_max_shares: float = Field(..., description="Max. shares of referenced articles")
    self_reference_avg_sharess: float = Field(..., description="Avg. shares of referenced articles")

    # Time features
    weekday_is_monday: float = Field(..., description="Was the article published on Monday?")
    weekday_is_tuesday: float = Field(..., description="Was the article published on Tuesday?")
    weekday_is_wednesday: float = Field(..., description="Was the article published on Wednesday?")
    weekday_is_thursday: float = Field(..., description="Was the article published on Thursday?")
    weekday_is_friday: float = Field(..., description="Was the article published on Friday?")
    weekday_is_saturday: float = Field(..., description="Was the article published on Saturday?")
    weekday_is_sunday: float = Field(..., description="Was the article published on Sunday?")
    is_weekend: float = Field(..., description="Was the article published on the weekend?")

    # LDA topic features
    LDA_00: float = Field(..., description="Closeness to LDA topic 0")
    LDA_01: float = Field(..., description="Closeness to LDA topic 1")
    LDA_02: float = Field(..., description="Closeness to LDA topic 2")
    LDA_03: float = Field(..., description="Closeness to LDA topic 3")
    LDA_04: float = Field(..., description="Closeness to LDA topic 4")

    # Sentiment and polarity features
    global_subjectivity: float = Field(..., description="Text subjectivity")
    global_sentiment_polarity: float = Field(..., description="Text sentiment polarity")
    global_rate_positive_words: float = Field(..., description="Rate of positive words in the content")
    global_rate_negative_words: float = Field(..., description="Rate of negative words in the content")
    rate_positive_words: float = Field(
        ..., description="Rate of positive words among non-neutral tokens"
    )
    rate_negative_words: float = Field(
        ..., description="Rate of negative words among non-neutral tokens"
    )
    avg_positive_polarity: float = Field(..., description="Avg. polarity of positive words")
    min_positive_polarity: float = Field(..., description="Min. polarity of positive words")
    max_positive_polarity: float = Field(..., description="Max. polarity of positive words")
    avg_negative_polarity: float = Field(..., description="Avg. polarity of negative words")
    min_negative_polarity: float = Field(..., description="Min. polarity of negative words")
    max_negative_polarity: float = Field(..., description="Max. polarity of negative words")

    # Title features
    title_subjectivity: float = Field(..., description="Title subjectivity")
    title_sentiment_polarity: float = Field(..., description="Title polarity")
    abs_title_subjectivity: float = Field(..., description="Absolute subjectivity level")
    abs_title_sentiment_polarity: float = Field(..., description="Absolute polarity level")

    # Mixed type column
    mixed_type_col: float = Field(..., description="Mixed type column")

    class Config:
        schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Response model for single prediction."""

    predicted_shares: int = Field(..., description="Predicted number of shares")
    log_prediction: float = Field(
        ..., description="Log-transformed prediction (model output)"
    )

    class Config:
        schema_extra = {
            "example": {
                "predicted_shares": 2500,
                "log_prediction": 7.824,
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""

    instances: List[NewsArticleFeatures] = Field(
        ...,
        description="List of news article features for batch prediction",
        min_items=1,
        max_items=1000,
    )

    @validator("instances")
    def validate_batch_size(cls, v):
        """Validate batch size."""
        if len(v) > 1000:
            raise ValueError("Batch size cannot exceed 1000 instances")
        return v


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""

    predictions: List[PredictionResponse] = Field(
        ..., description="List of predictions"
    )
    count: int = Field(..., description="Number of predictions")

    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {"predicted_shares": 2500, "log_prediction": 7.824},
                    {"predicted_shares": 1800, "log_prediction": 7.495},
                ],
                "count": 2,
            }
        }


class ModelInfo(BaseModel):
    """Model information and metadata."""

    status: str = Field(..., description="Model status (ready, not_initialized, error)")
    model_info: Dict[str, Any] = Field(..., description="Model metadata")
    features: Dict[str, Any] = Field(..., description="Feature information")
    target: str = Field(..., description="Target variable name")

    class Config:
        schema_extra = {
            "example": {
                "status": "ready",
                "model_info": {
                    "model_name": "RandomForestBase",
                    "load_strategy": "local",
                    "model_size_mb": 234.5,
                },
                "features": {
                    "count": 59,
                    "names": ["n_tokens_title", "n_tokens_content", "..."],
                },
                "target": "shares",
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status (healthy, unhealthy)")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: Optional[str] = Field(None, description="Loaded model name")
    version: str = Field(..., description="API version")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_name": "RandomForestBase",
                "version": "1.0.0",
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""

    detail: str = Field(..., description="Error message")
    error_type: Optional[str] = Field(None, description="Type of error")

    class Config:
        schema_extra = {
            "example": {
                "detail": "Missing required features: ['n_tokens_title']",
                "error_type": "ValidationError",
            }
        }
