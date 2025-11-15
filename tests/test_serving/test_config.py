"""
Unit tests for serving configuration module.

Tests environment variable loading, default values, and configuration validation.
"""

import pytest

from mlops_online_news_popularity.serving import config


class TestConfiguration:
    """Tests for serving configuration."""

    def test_default_model_name(self):
        """Test default model name."""
        assert config.MODEL_NAME == "RandomForestBase"

    def test_default_load_strategy(self):
        """Test default load strategy."""
        assert config.MODEL_LOAD_STRATEGY in ["local", "mlflow"]

    def test_api_host_default(self):
        """Test default API host."""
        assert config.API_HOST == "0.0.0.0"

    def test_api_port_default(self):
        """Test default API port."""
        assert config.API_PORT == 8000
        assert isinstance(config.API_PORT, int)

    def test_feature_names_count(self):
        """Test that there are exactly 59 features."""
        assert len(config.FEATURE_NAMES) == 59

    def test_feature_names_are_strings(self):
        """Test that all feature names are strings."""
        assert all(isinstance(name, str) for name in config.FEATURE_NAMES)

    def test_feature_names_no_duplicates(self):
        """Test that feature names are unique."""
        assert len(config.FEATURE_NAMES) == len(set(config.FEATURE_NAMES))

    def test_target_col(self):
        """Test target column name."""
        assert config.TARGET_COL == "shares"

    def test_log_level_default(self):
        """Test default log level."""
        assert config.LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class TestEnvironmentVariableOverrides:
    """Tests for environment variable overrides."""

    def test_model_name_override(self, monkeypatch):
        """Test MODEL_NAME can be overridden via env var."""
        import importlib

        monkeypatch.setenv("MODEL_NAME", "CustomModel")

        # Reload module to pick up new env var
        importlib.reload(config)

        assert config.MODEL_NAME == "CustomModel"

        # Cleanup - reload with original values
        monkeypatch.delenv("MODEL_NAME")
        importlib.reload(config)

    def test_api_port_override(self, monkeypatch):
        """Test API_PORT can be overridden via env var."""
        import importlib

        monkeypatch.setenv("API_PORT", "9000")

        # Reload module to pick up new env var
        importlib.reload(config)

        assert config.API_PORT == 9000

        # Cleanup
        monkeypatch.delenv("API_PORT")
        importlib.reload(config)

    def test_load_strategy_override(self, monkeypatch):
        """Test MODEL_LOAD_STRATEGY can be overridden via env var."""
        import importlib

        monkeypatch.setenv("MODEL_LOAD_STRATEGY", "mlflow")

        # Reload module to pick up new env var
        importlib.reload(config)

        assert config.MODEL_LOAD_STRATEGY == "mlflow"

        # Cleanup
        monkeypatch.delenv("MODEL_LOAD_STRATEGY")
        importlib.reload(config)


class TestFeatureNames:
    """Tests for feature name list."""

    def test_content_features_present(self):
        """Test that content features are in the list."""
        content_features = [
            "n_tokens_title",
            "n_tokens_content",
            "n_unique_tokens",
            "num_hrefs",
            "num_imgs"
        ]

        for feature in content_features:
            assert feature in config.FEATURE_NAMES

    def test_channel_features_present(self):
        """Test that channel features are in the list."""
        channel_features = [
            "data_channel_is_lifestyle",
            "data_channel_is_entertainment",
            "data_channel_is_bus",
            "data_channel_is_socmed",
            "data_channel_is_tech",
            "data_channel_is_world"
        ]

        for feature in channel_features:
            assert feature in config.FEATURE_NAMES

    def test_temporal_features_present(self):
        """Test that temporal features are in the list."""
        temporal_features = [
            "weekday_is_monday",
            "weekday_is_tuesday",
            "weekday_is_wednesday",
            "weekday_is_thursday",
            "weekday_is_friday",
            "weekday_is_saturday",
            "weekday_is_sunday",
            "is_weekend"
        ]

        for feature in temporal_features:
            assert feature in config.FEATURE_NAMES

    def test_lda_features_present(self):
        """Test that LDA topic features are in the list."""
        lda_features = ["LDA_00", "LDA_01", "LDA_02", "LDA_03", "LDA_04"]

        for feature in lda_features:
            assert feature in config.FEATURE_NAMES

    def test_sentiment_features_present(self):
        """Test that sentiment features are in the list."""
        sentiment_features = [
            "global_subjectivity",
            "global_sentiment_polarity",
            "title_subjectivity",
            "title_sentiment_polarity"
        ]

        for feature in sentiment_features:
            assert feature in config.FEATURE_NAMES
