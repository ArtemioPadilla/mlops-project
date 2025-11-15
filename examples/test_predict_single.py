"""
Example script to test single prediction endpoint.

Usage:
    python examples/test_predict_single.py
"""

import json

import requests

# API endpoint
API_URL = "http://localhost:8000"


def test_single_prediction():
    """Test single prediction endpoint."""
    print("=" * 70)
    print("Testing Single Prediction Endpoint")
    print("=" * 70)

    # Example article features (Business channel, Tuesday, moderate content)
    article_features = {
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

    # Make prediction request
    print("\nSending prediction request...")
    print(f"URL: {API_URL}/predict")

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=article_features,
            headers={"Content-Type": "application/json"},
        )

        # Check response
        if response.status_code == 200:
            result = response.json()
            print("\n✓ SUCCESS!")
            print("-" * 70)
            print(f"Predicted Shares: {result['predicted_shares']:,}")
            print(f"Log Prediction:   {result['log_prediction']:.4f}")
            print("-" * 70)
        else:
            print(f"\n✗ ERROR: Status code {response.status_code}")
            print(response.json())

    except requests.exceptions.ConnectionError:
        print("\n✗ ERROR: Could not connect to API server")
        print(f"Make sure the server is running at {API_URL}")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")


def test_health():
    """Test health endpoint."""
    print("\n" + "=" * 70)
    print("Testing Health Endpoint")
    print("=" * 70)

    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            result = response.json()
            print("\n✓ Server is healthy!")
            print(f"Status: {result['status']}")
            print(f"Model Loaded: {result['model_loaded']}")
            print(f"Model Name: {result.get('model_name', 'N/A')}")
            print(f"Version: {result['version']}")
        else:
            print(f"\n✗ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")


def test_model_info():
    """Test model info endpoint."""
    print("\n" + "=" * 70)
    print("Testing Model Info Endpoint")
    print("=" * 70)

    try:
        response = requests.get(f"{API_URL}/info")
        if response.status_code == 200:
            result = response.json()
            print("\n✓ Model info retrieved!")
            print(f"Status: {result['status']}")
            print(f"Model: {result['model_info'].get('model_name', 'N/A')}")
            print(f"Features: {result['features']['count']}")
            print(f"Target: {result['target']}")
        else:
            print(f"\n✗ Failed to get model info: {response.status_code}")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")


if __name__ == "__main__":
    # Run tests
    test_health()
    test_model_info()
    test_single_prediction()

    print("\n" + "=" * 70)
    print("Testing Complete!")
    print("=" * 70)
