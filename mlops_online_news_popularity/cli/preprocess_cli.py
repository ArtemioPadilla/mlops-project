"""
Preprocessing CLI for data preparation.

This script executes the complete model-agnostic preprocessing pipeline and saves
train/val/test splits to data/processed/ for use in model training.

Usage:
    python -m mlops_online_news_popularity.cli.preprocess_cli --help
    python -m mlops_online_news_popularity.cli.preprocess_cli --input data/raw/online_news_modified.csv
"""

import json
from pathlib import Path

from loguru import logger
import typer

from mlops_online_news_popularity.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from mlops_online_news_popularity.preprocessing import DataProcessor

app = typer.Typer()


@app.command()
def main(
    input_path: Path = typer.Option(
        RAW_DATA_DIR / "online_news_modified.csv",
        "--input",
        "-i",
        help="Path to raw CSV file",
    ),
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR,
        "--output-dir",
        "-o",
        help="Directory to save processed data",
    ),
    target_col: str = typer.Option(
        "shares",
        "--target",
        "-t",
        help="Name of target column",
    ),
    correlation_threshold: float = typer.Option(
        0.9,
        "--corr-threshold",
        help="Correlation threshold for feature removal",
    ),
):
    """
    Execute preprocessing pipeline and save splits to disk.

    This command:
    1. Loads and cleans raw data
    2. Engineers features (model-agnostic)
    3. Splits into train/val/test (70/15/15)
    4. Handles high correlation features
    5. Saves splits as CSV files
    6. Saves metadata as JSON
    """
    logger.info("=" * 70)
    logger.info("PREPROCESSING PIPELINE - CLI")
    logger.info("=" * 70)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Target column: {target_col}")

    # Ensure input exists
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        raise typer.Exit(code=1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Execute preprocessing pipeline
    logger.info("\nInitializing DataProcessor...")
    processor = DataProcessor(
        filepath=str(input_path),
        target_col=target_col,
        correlation_threshold=correlation_threshold,
    )

    logger.info("Running preprocessing pipeline...")
    processor.process()

    # Save splits
    logger.info("\n" + "=" * 70)
    logger.info("SAVING PROCESSED DATA")
    logger.info("=" * 70)

    splits = {
        "X_train": processor.X_train,
        "X_val": processor.X_val,
        "X_test": processor.X_test,
        "y_train": processor.y_train,
        "y_val": processor.y_val,
        "y_test": processor.y_test,
    }

    for name, df in splits.items():
        output_path = output_dir / f"{name}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {name}: {output_path} (shape: {df.shape})")

    # Save metadata
    metadata = {
        "target_col": target_col,
        "correlation_threshold": correlation_threshold,
        "cols_bin": processor.cols_bin,
        "cols_no_bin": processor.cols_no_bin,
        "cols_dropped_correlation": processor.cols_dropped_correlation,
        "numeric_features": processor.numeric_features,
        "final_feature_count": processor.X_train.shape[1],
        "splits": {
            "train_size": len(processor.X_train),
            "val_size": len(processor.X_val),
            "test_size": len(processor.X_test),
        },
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"\nSaved metadata: {metadata_path}")

    logger.info("\n" + "=" * 70)
    logger.success("PREPROCESSING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total features: {metadata['final_feature_count']}")
    logger.info(f"  Binary: {len(processor.cols_bin)}")
    logger.info(f"  Non-binary: {len(processor.cols_no_bin)}")
    logger.info(f"Train samples: {metadata['splits']['train_size']}")
    logger.info(f"Val samples: {metadata['splits']['val_size']}")
    logger.info(f"Test samples: {metadata['splits']['test_size']}")
    logger.info("=" * 70)


if __name__ == "__main__":
    app()
