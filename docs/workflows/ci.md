# Continuous Integration (CI) with GitHub Actions

This project uses GitHub Actions for automated testing and code quality checks on every push and pull request.

## CI Workflow Overview

The CI pipeline runs automatically when:
- Code is pushed to `main` or `develop` branches
- Pull requests are opened targeting `main`
- Manually triggered via GitHub Actions UI

## Jobs

### 1. Code Quality Checks (Lint)

Runs linting tools to ensure code quality and style consistency:

```bash
make lint
```

This executes:
- **flake8**: Checks for Python code issues
- **isort**: Verifies import sorting
- **black**: Ensures code formatting (99 char line length)

**Duration**: ~30 seconds

### 2. Tests

Runs the complete test suite with coverage reporting:

```bash
pip install -e .
pytest tests/ -v \
  --cov=mlops_online_news_popularity \
  --cov-report=xml \
  --cov-report=html \
  --cov-report=term-missing
```

**Test coverage includes**:
- Unit tests (preprocessing, modeling, serving)
- Integration tests (API endpoints, full pipeline)
- Async tests (FastAPI endpoints)

**Duration**: ~2-3 minutes

### 3. Test Summary

Aggregates results from all jobs and provides a summary.

## Artifacts

The CI workflow generates and uploads:

1. **Coverage Reports (XML)**: Uploaded to Codecov for tracking
2. **Coverage Reports (HTML)**: Available as workflow artifacts for 30 days

## Viewing CI Results

### GitHub Actions UI

1. Go to [Actions tab](https://github.com/ArtemioPadilla/mlops-project/actions)
2. Click on the workflow run
3. View job logs and test results

### Pull Request Status Checks

CI status is displayed directly on pull requests:
- ✅ Green checkmark: All checks passed
- ❌ Red X: Some checks failed
- 🟡 Yellow dot: Checks in progress

### Badges

CI status badges in README.md:

- ![CI](https://github.com/ArtemioPadilla/mlops-project/actions/workflows/ci.yml/badge.svg) - Shows current CI status
- ![Coverage](https://codecov.io/gh/ArtemioPadilla/mlops-project/branch/main/graph/badge.svg) - Shows code coverage percentage

## Local Testing

Run the same checks locally before pushing:

```bash
# Linting
make lint

# Tests with coverage
make test-coverage

# Or individual test suites
make test-unit           # Unit tests only
make test-integration    # Integration tests only
make test-serving        # Serving module tests
```

## Configuration Files

- **Workflow**: `.github/workflows/ci.yml`
- **Pytest Config**: `pyproject.toml` (lines 35-79)
- **Coverage Config**: `pyproject.toml` (coverage settings)
- **Linting Config**: `setup.cfg` (flake8, isort, black)

## Python Version Matrix

Currently testing on:
- **Python 3.10** (primary version)

To add more Python versions, edit `.github/workflows/ci.yml`:

```yaml
matrix:
  python-version: ['3.10', '3.11', '3.12']
```

## Codecov Integration (Optional)

Coverage reports are automatically uploaded to [Codecov](https://codecov.io).

### Setup (one-time)

1. Create account at [codecov.io](https://codecov.io)
2. Connect your GitHub repository
3. Add `CODECOV_TOKEN` to GitHub Secrets (Repository Settings > Secrets > Actions)

**Note**: Codecov integration is optional. The workflow will continue even if token is not configured (`fail_ci_if_error: false`).

## Troubleshooting

### CI Failing Locally Works

**Common causes**:
- Package not installed in editable mode: Run `pip install -e .`
- Missing dependencies: Run `pip install -r requirements.txt`
- Different Python version: CI uses 3.10, ensure you're using the same

**Solution**:
```bash
# Fresh environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
make test
```

### Linting Errors

Fix automatically with:

```bash
make format
```

This runs:
- `isort mlops_online_news_popularity`
- `black mlops_online_news_popularity`

### Coverage Below Threshold

Current coverage target: No minimum enforced

To add coverage requirements, edit `pyproject.toml`:

```toml
[tool.coverage.report]
fail_under = 80  # Fail if coverage below 80%
```

## Skipping CI

To skip CI on a commit (not recommended):

```bash
git commit -m "docs: update README [skip ci]"
```

## CI Performance

Typical run times:
- **Lint**: ~30 seconds
- **Tests**: ~2-3 minutes
- **Total**: ~3-4 minutes

**Optimization tips**:
- Cache pip dependencies (already enabled)
- Run jobs in parallel (already configured)
- Skip docs-only changes (already configured via `paths-ignore`)

## Future Enhancements

Potential additions:
- [ ] Security scanning (Bandit, Safety)
- [ ] Dependency updates (Dependabot)
- [ ] Performance benchmarks
- [ ] Docker image builds
- [ ] Deploy to staging environment

## Reproducibility Workflow

In addition to CI, the project has a **separate workflow** for testing reproducibility.

### Overview

**Workflow**: [`.github/workflows/reproducibility.yml`](../../.github/workflows/reproducibility.yml)

This workflow validates that the ML pipeline produces identical results across runs.

### When It Runs

- **Manual trigger**: Click "Run workflow" in GitHub Actions UI
- **Pull Requests to main**: Validates before merging important changes
- **Releases/tags**: Ensures published versions are reproducible
- **Weekly schedule**: (Optional, commented out) Monitors degradation

### What It Tests

1. Python 3.10 is used
2. Data preprocessing produces identical splits
3. Model training produces identical metrics
4. Model predictions are identical across runs

### Duration

~10-15 minutes per run

### Viewing Results

**In Pull Requests:**
- Automatic comment with test results
- Status check (required for merge, optional)

**In GitHub Actions:**
- Navigate to Actions → Reproducibility Test
- View detailed logs and artifacts

**Badges:**
- [![Reproducibility](https://github.com/ArtemioPadilla/mlops-project/actions/workflows/reproducibility.yml/badge.svg)](https://github.com/ArtemioPadilla/mlops-project/actions/workflows/reproducibility.yml)

### Manual Execution

```bash
# Trigger from GitHub UI
# Or run locally:
make test-reproducibility
```

### Configuration

To enable weekly monitoring, uncomment in `reproducibility.yml`:

```yaml
schedule:
  - cron: '0 9 * * 1'  # Mondays at 9 AM UTC
```

---

## Workflow Comparison

| Workflow | Purpose | Frequency | Duration |
|----------|---------|-----------|----------|
| **CI** ([`ci.yml`](../../.github/workflows/ci.yml)) | Tests + Lint | Every push/PR | ~3-4 min |
| **Reproducibility** ([`reproducibility.yml`](../../.github/workflows/reproducibility.yml)) | ML pipeline validation | PRs to main, releases | ~10-15 min |
| **Deploy Docs** | GitHub Pages deploy | Push to main | ~2 min |

---

## Related Documentation

- [Testing Guide](../TESTING_GUIDE.md)
- [Reproducibility Guide](../reproducibility.md)
- [Contributing Guidelines](../contributing.md)
- [Development Setup](../getting-started.md)

---

**CI Workflow**: [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)
**Reproducibility Workflow**: [`.github/workflows/reproducibility.yml`](../../.github/workflows/reproducibility.yml)
**Last Updated**: November 2024
