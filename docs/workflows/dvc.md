# DVC Data Versioning

Using DVC to version datasets.

## Initial Setup

```bash
# Initialize DVC (already done)
dvc init

# Add remote storage (optional)
dvc remote add -d myremote /path/to/dvc-storage
```

## Tracking Data

```bash
# Track raw data
dvc add data/raw/online_news_modified.csv

# Commit DVC file to git
git add data/raw/online_news_modified.csv.dvc data/raw/.gitignore
git commit -m "Track raw data with DVC"

# Track processed data
dvc add data/processed/
git add data/processed.dvc
git commit -m "Track processed data"
```

## Pulling Data

```bash
# Pull latest data
dvc pull

# Pull specific version
git checkout <commit-hash>
dvc checkout
```

## Configuration

See `.dvc/config` for remote storage settings.
