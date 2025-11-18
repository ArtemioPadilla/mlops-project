#!/usr/bin/env python3
"""
Create a small sample dataset for CI testing.

This script creates a subset of the full dataset (~2000 rows) that can be
committed to git for use in GitHub Actions CI. The sample maintains the
distribution characteristics of the full dataset.

Usage:
    python scripts/create_sample_data.py

Output:
    data/sample/online_news_sample.csv (~500KB, safe to commit)
"""
import sys
from pathlib import Path

import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "online_news_modified.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "sample"
OUTPUT_PATH = OUTPUT_DIR / "online_news_sample.csv"

# Sampling parameters
SAMPLE_SIZE = 2000
RANDOM_STATE = 42


def main():
    print("=" * 60)
    print("Creating Sample Dataset for CI")
    print("=" * 60)
    print()

    # Check input file exists
    if not INPUT_PATH.exists():
        print(f"❌ Error: Input file not found: {INPUT_PATH}")
        print()
        print("Please ensure the full dataset exists at:")
        print(f"  {INPUT_PATH}")
        sys.exit(1)

    # Read full dataset
    print(f"📖 Reading full dataset from:")
    print(f"   {INPUT_PATH}")
    df_full = pd.read_csv(INPUT_PATH)
    print(f"✅ Loaded {len(df_full):,} rows, {len(df_full.columns)} columns")
    print()

    # Sample data with fixed random state
    print(f"🎲 Sampling {SAMPLE_SIZE:,} rows (random_state={RANDOM_STATE})...")

    # Use simple random sampling (not stratified) for simplicity
    # This is sufficient for CI reproducibility testing
    df_sample = df_full.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

    # Reset index to have clean sequential indices
    df_sample = df_sample.reset_index(drop=True)

    print(f"✅ Sample created: {len(df_sample):,} rows")
    print()

    # Show statistics comparison
    print("📊 Statistics Comparison:")
    print("-" * 60)

    target_col = "shares"
    if target_col in df_full.columns:
        # Convert target to numeric (in case it's stored as string)
        df_full_target = pd.to_numeric(df_full[target_col], errors='coerce')
        df_sample_target = pd.to_numeric(df_sample[target_col], errors='coerce')

        print(f"Target column: {target_col}")
        print()
        print(f"{'Metric':<20} {'Full Dataset':>15} {'Sample':>15}")
        print("-" * 60)
        print(f"{'Rows':<20} {len(df_full):>15,} {len(df_sample):>15,}")
        print(f"{'Mean':<20} {df_full_target.mean():>15,.2f} {df_sample_target.mean():>15,.2f}")
        print(f"{'Median':<20} {df_full_target.median():>15,.2f} {df_sample_target.median():>15,.2f}")
        print(f"{'Std Dev':<20} {df_full_target.std():>15,.2f} {df_sample_target.std():>15,.2f}")
        print(f"{'Min':<20} {df_full_target.min():>15,.0f} {df_sample_target.min():>15,.0f}")
        print(f"{'Max':<20} {df_full_target.max():>15,.0f} {df_sample_target.max():>15,.0f}")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save sample dataset
    print(f"💾 Saving sample dataset to:")
    print(f"   {OUTPUT_PATH}")
    df_sample.to_csv(OUTPUT_PATH, index=False)

    # Get file size
    file_size_bytes = OUTPUT_PATH.stat().st_size
    file_size_kb = file_size_bytes / 1024
    file_size_mb = file_size_kb / 1024

    print(f"✅ Sample saved successfully!")
    print(f"   Size: {file_size_kb:.1f} KB ({file_size_mb:.2f} MB)")
    print()

    print("=" * 60)
    print("✅ Sample Dataset Created Successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Add this file to git:")
    print(f"     git add {OUTPUT_PATH}")
    print("  2. Commit the sample data:")
    print('     git commit -m "data: Add sample dataset for CI testing"')
    print()
    print("This sample dataset will be used in GitHub Actions CI")
    print("for reproducibility testing, while local development")
    print("continues to use the full dataset.")
    print()


if __name__ == "__main__":
    main()
