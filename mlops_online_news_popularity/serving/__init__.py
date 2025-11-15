"""
FastAPI serving module for Online News Popularity prediction.

This module provides a REST API for serving trained ML models with support
for both online (single) and batch predictions.
"""

from mlops_online_news_popularity.serving.model_handler import ModelHandler
from mlops_online_news_popularity.serving.app import app

__all__ = ["ModelHandler", "app"]
