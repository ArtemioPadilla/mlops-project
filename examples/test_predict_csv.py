"""
Example script to test batch prediction endpoint (CSV upload).

Usage:
    python examples/test_predict_csv.py
"""

import requests

# API endpoint
API_URL = "http://localhost:8000"


def test_csv_prediction():
    """Test CSV batch prediction endpoint."""
    print("=" * 70)
    print("Testing Batch Prediction Endpoint (CSV Upload)")
    print("=" * 70)

    # Path to sample CSV file
    csv_file_path = "examples/sample_data.csv"

    print(f"\nUploading CSV file: {csv_file_path}")
    print(f"URL: {API_URL}/predict/batch/csv")

    try:
        # Open and upload CSV file
        with open(csv_file_path, "rb") as f:
            files = {"file": ("sample_data.csv", f, "text/csv")}
            response = requests.post(f"{API_URL}/predict/batch/csv", files=files)

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

    except FileNotFoundError:
        print(f"\n✗ ERROR: File not found: {csv_file_path}")
        print(
            "Please create sample_data.csv using the provided sample or from processed data"
        )
    except requests.exceptions.ConnectionError:
        print("\n✗ ERROR: Could not connect to API server")
        print(f"Make sure the server is running at {API_URL}")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")


if __name__ == "__main__":
    test_csv_prediction()

    print("\n" + "=" * 70)
    print("Testing Complete!")
    print("=" * 70)
