#!/bin/bash
# Test reproducibility by running the ML pipeline twice and comparing outputs
#
# This script validates that:
# 1. Data preprocessing produces identical train/val/test splits
# 2. Model training produces identical metrics and predictions
# 3. Results are deterministic across runs

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# Temporary directories for comparison
RUN1_DIR="${PROJECT_ROOT}/.reproducibility_test/run1"
RUN2_DIR="${PROJECT_ROOT}/.reproducibility_test/run2"

echo "==========================================="
echo "REPRODUCIBILITY TEST"
echo "==========================================="
echo ""

# Cleanup function
cleanup() {
    echo -e "${YELLOW}Cleaning up test directories...${NC}"
    rm -rf .reproducibility_test
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Step 1: Verify Python version
echo -e "${BLUE}Step 1: Verifying Python version...${NC}"

# Try multiple methods to get Python version
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo -e "${RED}❌ Python not found!${NC}"
    echo "Please install Python 3.10"
    exit 1
fi

# Get Python version using python itself (more reliable)
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)

if [ -z "$PYTHON_VERSION" ]; then
    echo -e "${RED}❌ Could not determine Python version!${NC}"
    exit 1
fi

PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" != "3" ] || [ "$PYTHON_MINOR" != "10" ]; then
    echo -e "${RED}❌ Python version mismatch!${NC}"
    echo "Expected: 3.10.x"
    echo "Got: $PYTHON_VERSION"
    echo ""
    echo "Please activate a Python 3.10 environment:"
    echo "  python3.10 -m venv venv"
    echo "  source venv/bin/activate"
    exit 1
fi

echo -e "${GREEN}✅ Python $PYTHON_VERSION detected${NC}"
echo ""

# Step 2: Verify package installation
echo -e "${BLUE}Step 2: Verifying package installation...${NC}"
if ! $PYTHON_CMD -c "import mlops_online_news_popularity" 2>/dev/null; then
    echo -e "${RED}❌ Package not installed!${NC}"
    echo "Run: pip install -e ."
    exit 1
fi
echo -e "${GREEN}✅ Package installed${NC}"
echo ""

# Step 3: Create test directories
echo -e "${BLUE}Step 3: Setting up test directories...${NC}"
mkdir -p "$RUN1_DIR" "$RUN2_DIR"
echo -e "${GREEN}✅ Test directories created${NC}"
echo ""

# Step 4: Run pipeline (Run 1)
echo -e "${BLUE}Step 4: Running pipeline (Run 1)...${NC}"
echo "Preprocessing..."

# Clean previous runs
rm -rf data/processed/*

# Run preprocessing
$PYTHON_CMD -m mlops_online_news_popularity.cli.preprocess_cli \
    --input data/raw/online_news_modified.csv \
    --output-dir data/processed \
    --corr-threshold 0.9 \
    > /dev/null 2>&1

# Save preprocessed data
cp -r data/processed/* "$RUN1_DIR/"

echo "Training model (Ridge)..."
# Run training
$PYTHON_CMD -m mlops_online_news_popularity.cli.train_cli train-single --model ridge \
    > "${RUN1_DIR}/train_output.txt" 2>&1

# Extract metrics from output
RUN1_TRAIN_RMSE=$(grep "Train RMSE" "${RUN1_DIR}/train_output.txt" | awk '{print $3}' | head -1)
RUN1_VAL_RMSE=$(grep "Val RMSE" "${RUN1_DIR}/train_output.txt" | awk '{print $3}' | head -1)
RUN1_TEST_RMSE=$(grep "Test RMSE" "${RUN1_DIR}/train_output.txt" | awk '{print $3}' | head -1)

# Save model file
LATEST_MODEL=$(ls -t models/ridge_best_*.pkl 2>/dev/null | head -1)
if [ -n "$LATEST_MODEL" ]; then
    cp "$LATEST_MODEL" "${RUN1_DIR}/model.pkl"
fi

echo -e "${GREEN}✅ Run 1 completed${NC}"
echo "  Train RMSE: $RUN1_TRAIN_RMSE"
echo "  Val RMSE: $RUN1_VAL_RMSE"
echo "  Test RMSE: $RUN1_TEST_RMSE"
echo ""

# Step 5: Run pipeline (Run 2)
echo -e "${BLUE}Step 5: Running pipeline (Run 2)...${NC}"
echo "Cleaning intermediate files..."

# Clean all outputs
rm -rf data/processed/*
rm -f models/ridge_best_*.pkl

echo "Preprocessing..."
# Run preprocessing again
$PYTHON_CMD -m mlops_online_news_popularity.cli.preprocess_cli \
    --input data/raw/online_news_modified.csv \
    --output-dir data/processed \
    --corr-threshold 0.9 \
    > /dev/null 2>&1

# Save preprocessed data
cp -r data/processed/* "$RUN2_DIR/"

echo "Training model (Ridge)..."
# Run training again
$PYTHON_CMD -m mlops_online_news_popularity.cli.train_cli train-single --model ridge \
    > "${RUN2_DIR}/train_output.txt" 2>&1

# Extract metrics
RUN2_TRAIN_RMSE=$(grep "Train RMSE" "${RUN2_DIR}/train_output.txt" | awk '{print $3}' | head -1)
RUN2_VAL_RMSE=$(grep "Val RMSE" "${RUN2_DIR}/train_output.txt" | awk '{print $3}' | head -1)
RUN2_TEST_RMSE=$(grep "Test RMSE" "${RUN2_DIR}/train_output.txt" | awk '{print $3}' | head -1)

# Save model file
LATEST_MODEL=$(ls -t models/ridge_best_*.pkl 2>/dev/null | head -1)
if [ -n "$LATEST_MODEL" ]; then
    cp "$LATEST_MODEL" "${RUN2_DIR}/model.pkl"
fi

echo -e "${GREEN}✅ Run 2 completed${NC}"
echo "  Train RMSE: $RUN2_TRAIN_RMSE"
echo "  Val RMSE: $RUN2_VAL_RMSE"
echo "  Test RMSE: $RUN2_TEST_RMSE"
echo ""

# Step 6: Compare results
echo -e "${BLUE}Step 6: Comparing results...${NC}"
echo ""

FAILURES=0

# Compare data splits (row counts)
echo "Comparing data splits..."
RUN1_TRAIN_ROWS=$(wc -l < "${RUN1_DIR}/X_train.csv")
RUN2_TRAIN_ROWS=$(wc -l < "${RUN2_DIR}/X_train.csv")
RUN1_VAL_ROWS=$(wc -l < "${RUN1_DIR}/X_val.csv")
RUN2_VAL_ROWS=$(wc -l < "${RUN2_DIR}/X_val.csv")
RUN1_TEST_ROWS=$(wc -l < "${RUN1_DIR}/X_test.csv")
RUN2_TEST_ROWS=$(wc -l < "${RUN2_DIR}/X_test.csv")

if [ "$RUN1_TRAIN_ROWS" -eq "$RUN2_TRAIN_ROWS" ] && \
   [ "$RUN1_VAL_ROWS" -eq "$RUN2_VAL_ROWS" ] && \
   [ "$RUN1_TEST_ROWS" -eq "$RUN2_TEST_ROWS" ]; then
    echo -e "${GREEN}✅ Data splits match${NC}"
    echo "  Train: $RUN1_TRAIN_ROWS rows"
    echo "  Val: $RUN1_VAL_ROWS rows"
    echo "  Test: $RUN1_TEST_ROWS rows"
else
    echo -e "${RED}❌ Data splits differ!${NC}"
    echo "  Train: $RUN1_TRAIN_ROWS vs $RUN2_TRAIN_ROWS"
    echo "  Val: $RUN1_VAL_ROWS vs $RUN2_VAL_ROWS"
    echo "  Test: $RUN1_TEST_ROWS vs $RUN2_TEST_ROWS"
    FAILURES=$((FAILURES + 1))
fi
echo ""

# Compare metrics
echo "Comparing metrics..."
METRICS_MATCH=1

if [ "$RUN1_TRAIN_RMSE" != "$RUN2_TRAIN_RMSE" ]; then
    echo -e "${RED}❌ Train RMSE differs: $RUN1_TRAIN_RMSE vs $RUN2_TRAIN_RMSE${NC}"
    METRICS_MATCH=0
    FAILURES=$((FAILURES + 1))
fi

if [ "$RUN1_VAL_RMSE" != "$RUN2_VAL_RMSE" ]; then
    echo -e "${RED}❌ Val RMSE differs: $RUN1_VAL_RMSE vs $RUN2_VAL_RMSE${NC}"
    METRICS_MATCH=0
    FAILURES=$((FAILURES + 1))
fi

if [ "$RUN1_TEST_RMSE" != "$RUN2_TEST_RMSE" ]; then
    echo -e "${RED}❌ Test RMSE differs: $RUN1_TEST_RMSE vs $RUN2_TEST_RMSE${NC}"
    METRICS_MATCH=0
    FAILURES=$((FAILURES + 1))
fi

if [ $METRICS_MATCH -eq 1 ]; then
    echo -e "${GREEN}✅ Metrics match exactly${NC}"
    echo "  Train RMSE: $RUN1_TRAIN_RMSE"
    echo "  Val RMSE: $RUN1_VAL_RMSE"
    echo "  Test RMSE: $RUN1_TEST_RMSE"
fi
echo ""

# Compare model files (via predictions on same data)
if [ -f "${RUN1_DIR}/model.pkl" ] && [ -f "${RUN2_DIR}/model.pkl" ]; then
    echo "Comparing model predictions..."

    # Create a simple Python script to compare predictions
    $PYTHON_CMD - <<EOF
import joblib
import pandas as pd
import numpy as np

model1 = joblib.load("${RUN1_DIR}/model.pkl")
model2 = joblib.load("${RUN2_DIR}/model.pkl")

X_test = pd.read_csv("${RUN1_DIR}/X_test.csv")

preds1 = model1.predict(X_test)
preds2 = model2.predict(X_test)

if np.allclose(preds1, preds2, atol=1e-10):
    print("✅ Model predictions identical")
    exit(0)
else:
    max_diff = np.max(np.abs(preds1 - preds2))
    print(f"❌ Model predictions differ (max diff: {max_diff})")
    exit(1)
EOF

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Model predictions identical${NC}"
    else
        echo -e "${RED}❌ Model predictions differ${NC}"
        FAILURES=$((FAILURES + 1))
    fi
fi
echo ""

# Final result
echo "==========================================="
if [ $FAILURES -eq 0 ]; then
    echo -e "${GREEN}✅ REPRODUCIBILITY TEST PASSED${NC}"
    echo "==========================================="
    echo ""
    echo "Summary:"
    echo "  • Data splits are identical"
    echo "  • Metrics are identical"
    echo "  • Model predictions are identical"
    echo ""
    echo "The ML pipeline is fully reproducible! 🎉"
    echo ""
    exit 0
else
    echo -e "${RED}❌ REPRODUCIBILITY TEST FAILED${NC}"
    echo "==========================================="
    echo ""
    echo "$FAILURES check(s) failed"
    echo ""
    echo "Possible causes:"
    echo "  1. Random seed not set correctly"
    echo "  2. Different package versions"
    echo "  3. Non-deterministic operations"
    echo ""
    echo "See docs/reproducibility.md for troubleshooting"
    echo ""
    exit 1
fi
