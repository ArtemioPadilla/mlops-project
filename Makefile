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
	$(PYTHON_INTERPRETER) -m pytest tests

## Run tests with coverage report
.PHONY: test-coverage
test-coverage:
	$(PYTHON_INTERPRETER) -m pytest tests --cov --cov-report=html --cov-report=term

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
