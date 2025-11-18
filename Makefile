#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mlops-project
PYTHON_VERSION = 3.10

ifeq ($(OS),Windows_NT)
    PYTHON_INTERPRETER = python
else
    PYTHON_INTERPRETER = python3.10
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 mlops_online_news_popularity
	isort --check --diff mlops_online_news_popularity
	black --check mlops_online_news_popularity

## Format source code with black
.PHONY: format
format:
	isort mlops_online_news_popularity
	black mlops_online_news_popularity



## Run all tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest -v tests

## Run tests with coverage report
.PHONY: test-coverage
test-coverage:
	$(PYTHON_INTERPRETER) -m pytest -v tests --cov --cov-report=html --cov-report=term

## Run only serving module tests
.PHONY: test-serving
test-serving:
	$(PYTHON_INTERPRETER) -m pytest tests/test_serving -v

## Run only unit tests
.PHONY: test-unit
test-unit:
	$(PYTHON_INTERPRETER) -m pytest tests -m unit -v

## Run only integration tests
.PHONY: test-integration
test-integration:
	$(PYTHON_INTERPRETER) -m pytest tests -m integration -v


#################################################################################
# REPRODUCIBILITY TESTS                                                         #
#################################################################################

## Check Python version (must be 3.10 for reproducibility)
.PHONY: check-python
check-python:
	@echo "Checking Python version..."
	@PYTHON_VERSION=$$($(PYTHON_INTERPRETER) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "unknown"); \
	if [ "$$PYTHON_VERSION" != "3.10" ]; then \
		echo "❌ Python version mismatch!"; \
		echo "   Expected: 3.10.x"; \
		echo "   Got: $$PYTHON_VERSION"; \
		echo ""; \
		echo "Solutions:"; \
		echo "  1. Activate Python 3.10 environment:"; \
		echo "     python3.10 -m venv venv310"; \
		echo "     source venv310/bin/activate  # or venv310\\Scripts\\activate on Windows"; \
		echo ""; \
		echo "  2. Or run dev mode (not recommended for official validation):"; \
		echo "     make test-reproducibility-dev"; \
		echo ""; \
		echo "  3. Or use Docker (guaranteed Python 3.10):"; \
		echo "     make test-reproducibility-docker"; \
		exit 1; \
	else \
		echo "✅ Python $$PYTHON_VERSION detected"; \
	fi

## Test reproducibility (strict - requires Python 3.10)
.PHONY: test-reproducibility
test-reproducibility: check-python
	@echo "Running reproducibility test (strict mode)..."
	@bash scripts/test_reproducibility.sh

## Test reproducibility in dev mode (allows any Python 3.x)
.PHONY: test-reproducibility-dev
test-reproducibility-dev:
	@echo "⚠️  Running reproducibility test in DEV mode (version check disabled)"
	@echo "Note: Results may differ from production if not using Python 3.10"
	@echo ""
	@sed 's/if \[ "$$PYTHON_MAJOR" != "3" \] || \[ "$$PYTHON_MINOR" != "10" \]; then/if false; then/' \
		scripts/test_reproducibility.sh | bash

## Test reproducibility using Docker (guaranteed Python 3.10)
.PHONY: test-reproducibility-docker
test-reproducibility-docker:
	@echo "Running reproducibility test in Docker (Python 3.10)..."
	@echo "Note: First run will take 5-10 minutes to install dependencies"
	@echo ""
	@docker run -it --rm \
		-v $$(pwd):/app \
		-w /app \
		python:3.10-slim \
		bash -c "echo '📦 Installing dependencies from requirements.txt...' && \
		         pip install --no-cache-dir -r requirements.txt && \
		         echo '' && \
		         echo '📦 Installing package in editable mode...' && \
		         pip install -e . && \
		         echo '' && \
		         echo '🧪 Running reproducibility test...' && \
		         bash scripts/test_reproducibility.sh"

## Quick reproducibility test (metrics only, faster)
.PHONY: test-reproducibility-quick
test-reproducibility-quick:
	@echo "Running quick reproducibility test (metrics comparison only)..."
	@mkdir -p .repro_quick
	@echo "Run 1..."
	@$(PYTHON_INTERPRETER) -m mlops_online_news_popularity.cli.preprocess_cli > /dev/null 2>&1
	@$(PYTHON_INTERPRETER) -m mlops_online_news_popularity.cli.train_cli train-single --model ridge > .repro_quick/run1.log 2>&1
	@grep "Test RMSE" .repro_quick/run1.log | head -1 > .repro_quick/metrics.txt
	@echo "Cleaning..."
	@rm -rf data/processed/* models/ridge_best_*.pkl
	@echo "Run 2..."
	@$(PYTHON_INTERPRETER) -m mlops_online_news_popularity.cli.preprocess_cli > /dev/null 2>&1
	@$(PYTHON_INTERPRETER) -m mlops_online_news_popularity.cli.train_cli train-single --model ridge > .repro_quick/run2.log 2>&1
	@grep "Test RMSE" .repro_quick/run2.log | head -1 >> .repro_quick/metrics.txt
	@echo ""
	@echo "Results:"
	@cat .repro_quick/metrics.txt
	@if diff <(head -1 .repro_quick/metrics.txt) <(tail -1 .repro_quick/metrics.txt) > /dev/null; then \
		echo "✅ Metrics match - Pipeline is reproducible!"; \
	else \
		echo "❌ Metrics differ - Check random seeds and dependencies"; \
	fi
	@rm -rf .repro_quick


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@echo "Creating virtual environment..."
ifeq ($(OS),Windows_NT)
	@$(PYTHON_INTERPRETER) -m venv venv
	@echo ">>> Virtual environment created. Activate with: venv\Scripts\activate"
else
	@$(PYTHON_INTERPRETER) -m venv venv
	@echo ">>> Virtual environment created. Activate with: source venv/bin/activate"
endif
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Run preprocessing pipeline (creates train/val/test splits)
.PHONY: preprocess
preprocess:
	$(PYTHON_INTERPRETER) -m mlops_online_news_popularity.cli.preprocess_cli

## Train and compare multiple models from config
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m mlops_online_news_popularity.cli.train_cli train-compare config/models.yaml

## Train a single model with MLflow tracking (for quick testing)
.PHONY: train-single
train-single:
	$(PYTHON_INTERPRETER) -m mlops_online_news_popularity.cli.train_cli train-single

## Run complete MLOps pipeline (preprocess + train)
.PHONY: pipeline
pipeline: preprocess train

## Start MLflow UI server
.PHONY: mlflow-ui
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow_artifacts/dev/mlflow.db --port 5001

## Run FastAPI server locally (development)
.PHONY: serve
serve:
	$(PYTHON_INTERPRETER) -m uvicorn mlops_online_news_popularity.serving.app:app --reload --host 0.0.0.0 --port 8000

## Run FastAPI server (production)
.PHONY: serve-prod
serve-prod:
	$(PYTHON_INTERPRETER) -m uvicorn mlops_online_news_popularity.serving.app:app --host 0.0.0.0 --port 8000 --workers 4

## Build Docker image
.PHONY: docker-build
docker-build:
	bash scripts/docker_build.sh

## Run Docker container
.PHONY: docker-run
docker-run:
	bash scripts/docker_run.sh

## Start services with docker-compose
.PHONY: docker-up
docker-up:
	docker compose up -d

## Stop docker-compose services
.PHONY: docker-down
docker-down:
	docker compose down

## View docker-compose logs
.PHONY: docker-logs
docker-logs:
	@if docker compose ps 2>/dev/null | grep -q online-news-predictor; then \
		docker compose logs -f; \
	elif docker ps --format '{{.Names}}' | grep -q '^online-news-predictor$$'; then \
		echo "Container not managed by docker-compose, using docker logs..."; \
		docker logs -f online-news-predictor; \
	else \
		echo "Error: Container 'online-news-predictor' not found"; \
		echo "Start the container with 'make docker-run' or 'make docker-up'"; \
		exit 1; \
	fi

## Tag Docker image for DockerHub
.PHONY: docker-tag
docker-tag:
	@echo "Tagging image for DockerHub..."
	@VERSION=$$(git describe --tags --always --dirty 2>/dev/null || echo "dev"); \
	docker tag ml-service:latest artemiop/mlops-news-predictor:latest; \
	docker tag ml-service:latest artemiop/mlops-news-predictor:$$VERSION; \
	echo "Tagged as:"; \
	echo "  - artemiop/mlops-news-predictor:latest"; \
	echo "  - artemiop/mlops-news-predictor:$$VERSION"

## Push Docker image to DockerHub
.PHONY: docker-push
docker-push: docker-tag
	@echo "Pushing images to DockerHub..."
	@VERSION=$$(git describe --tags --always --dirty 2>/dev/null || echo "dev"); \
	docker push artemiop/mlops-news-predictor:latest; \
	docker push artemiop/mlops-news-predictor:$$VERSION; \
	echo "Pushed successfully!"

## Build, tag, and push Docker image to DockerHub
.PHONY: docker-release
docker-release: docker-build docker-push
	@VERSION=$$(git describe --tags --always --dirty 2>/dev/null || echo "dev"); \
	echo "Released artemiop/mlops-news-predictor:$$VERSION to DockerHub"

## Test API endpoints (single prediction)
.PHONY: test-api
test-api:
	$(PYTHON_INTERPRETER) examples/test_predict_single.py

## Test API batch prediction (JSON)
.PHONY: test-api-batch
test-api-batch:
	$(PYTHON_INTERPRETER) examples/test_predict_batch.py

## Test API batch prediction (CSV)
.PHONY: test-api-csv
test-api-csv:
	$(PYTHON_INTERPRETER) examples/test_predict_csv.py

## Build documentation
.PHONY: docs
docs:
	mkdocs build

## Serve documentation locally
.PHONY: docs-serve
docs-serve:
	mkdocs serve

## Deploy documentation to GitHub Pages
.PHONY: docs-deploy
docs-deploy:
	mkdocs gh-deploy --force


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
