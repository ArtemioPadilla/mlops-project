#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mlops-project
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python3.10

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



## Run tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



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
	mlflow ui --backend-store-uri sqlite:///mlflow/dev/mlflow.db --port 5001

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
