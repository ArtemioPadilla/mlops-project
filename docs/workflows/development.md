# Development Workflow

Iterative development workflow for model experimentation.

## Typical Workflow

1. **Modify configuration**
2. **Run preprocessing** (if data changes)
3. **Train models**
4. **Evaluate in MLflow UI**
5. **Iterate**

## Example Session

```bash
# 1. Edit model configuration
vim config/models.yaml

# 2. Train with new config
make train

# 3. View results
make mlflow-ui

# 4. If good, commit changes
git add config/models.yaml
git commit -m "feat: add XGBoost model"
```

## Quick Iteration

For fast experimentation, train a single model:

```bash
make train-single  # Default: Ridge

# Or specify model
python -m mlops_online_news_popularity.cli.train_cli train-single --model randomforest
```

## Code Changes

After modifying preprocessing or modeling code:

```bash
# 1. Format code
make format

# 2. Run tests
make test

# 3. Re-run pipeline
make pipeline
```
