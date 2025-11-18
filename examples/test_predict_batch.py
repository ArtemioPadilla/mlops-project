"""
Example script to test batch prediction endpoint (JSON).

Usage:
    python examples/test_predict_batch.py
"""

import requests

# API endpoint
API_URL = "http://localhost:8000"


def create_sample_article(channel: str, day: str, content_size: int = 500) -> dict:
    """Create a sample article with specified characteristics."""
    # Channel flags
    channels = {
        "lifestyle": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "entertainment": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "business": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "socmed": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "tech": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "world": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    }

    # Day flags
    days = {
        "monday": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "tuesday": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "wednesday": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "thursday": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "friday": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "saturday": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "sunday": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    }

    channel_flags = channels.get(channel.lower(), channels["business"])
    day_flags = days.get(day.lower(), days["monday"])
    is_weekend = 1.0 if day.lower() in ["saturday", "sunday"] else 0.0

    return {
        "n_tokens_title": 10.0,
        "n_tokens_content": float(content_size),
        "n_unique_tokens": 0.5,
        "n_non_stop_words": 0.8,
        "n_non_stop_unique_tokens": 0.6,
        "num_hrefs": 10.0,
        "num_self_hrefs": 2.0,
        "num_imgs": 5.0,
        "num_videos": 1.0,
        "average_token_length": 4.5,
        "num_keywords": 7.0,
        "data_channel_is_lifestyle": channel_flags[0],
        "data_channel_is_entertainment": channel_flags[1],
        "data_channel_is_bus": channel_flags[2],
        "data_channel_is_socmed": channel_flags[3],
        "data_channel_is_tech": channel_flags[4],
        "data_channel_is_world": channel_flags[5],
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
        "weekday_is_monday": day_flags[0],
        "weekday_is_tuesday": day_flags[1],
        "weekday_is_wednesday": day_flags[2],
        "weekday_is_thursday": day_flags[3],
        "weekday_is_friday": day_flags[4],
        "weekday_is_saturday": day_flags[5],
        "weekday_is_sunday": day_flags[6],
        "is_weekend": is_weekend,
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


def test_batch_prediction():
    """Test batch prediction endpoint."""
    print("=" * 70)
    print("Testing Batch Prediction Endpoint (JSON)")
    print("=" * 70)

    # Create batch of sample articles
    articles = [
        create_sample_article("tech", "monday", 800),
        create_sample_article("entertainment", "friday", 600),
        create_sample_article("business", "tuesday", 1000),
        create_sample_article("lifestyle", "sunday", 400),
        create_sample_article("world", "wednesday", 700),
    ]

    batch_request = {"instances": articles}

    # Make batch prediction request
    print(f"\nSending batch prediction request for {len(articles)} articles...")
    print(f"URL: {API_URL}/predict/batch")

    try:
        response = requests.post(
            f"{API_URL}/predict/batch",
            json=batch_request,
            headers={"Content-Type": "application/json"},
        )

        # Check response
        if response.status_code == 200:
            result = response.json()
            print("\n✓ SUCCESS!")
            print("-" * 70)
            print(f"Total Predictions: {result['count']}")
            print("\nPredictions:")
            print(f"{'#':<5} {'Predicted Shares':<20} {'Log Prediction':<15}")
            print("-" * 70)

            for i, pred in enumerate(result["predictions"], 1):
                print(
                    f"{i:<5} {pred['predicted_shares']:>18,}  "
                    f"{pred['log_prediction']:>14.4f}"
                )

            print("-" * 70)

            # Calculate statistics
            shares = [p["predicted_shares"] for p in result["predictions"]]
            print(f"\nStatistics:")
            print(f"  Min:  {min(shares):>8,} shares")
            print(f"  Max:  {max(shares):>8,} shares")
            print(f"  Avg:  {sum(shares)//len(shares):>8,} shares")
        else:
            print(f"\n✗ ERROR: Status code {response.status_code}")
            print(response.json())

    except requests.exceptions.ConnectionError:
        print("\n✗ ERROR: Could not connect to API server")
        print(f"Make sure the server is running at {API_URL}")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")


if __name__ == "__main__":
    test_batch_prediction()

    print("\n" + "=" * 70)
    print("Testing Complete!")
    print("=" * 70)
