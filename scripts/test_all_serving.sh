#!/bin/bash
# Complete serving test script
# Tests: unit tests, coverage, API endpoints (if server is running)

set -e  # Exit on error

echo "=========================================="
echo "MLOps Project - Serving Complete Test"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Unit Tests
echo "${YELLOW}[1/4] Running Unit Tests...${NC}"
echo "----------------------------------------"
python3.10 -m pytest tests/test_serving -v --tb=short
if [ $? -eq 0 ]; then
    echo "${GREEN}✓ Unit tests passed!${NC}"
else
    echo "${RED}✗ Unit tests failed!${NC}"
    exit 1
fi
echo ""

# 2. Coverage Report
echo "${YELLOW}[2/4] Generating Coverage Report...${NC}"
echo "----------------------------------------"
python3.10 -m pytest tests/test_serving --cov --cov-report=term --cov-report=html
if [ $? -eq 0 ]; then
    echo "${GREEN}✓ Coverage report generated: htmlcov/index.html${NC}"
else
    echo "${RED}✗ Coverage generation failed!${NC}"
    exit 1
fi
echo ""

# 3. Check if API server is running
echo "${YELLOW}[3/4] Checking API Server...${NC}"
echo "----------------------------------------"
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "${GREEN}✓ API server is running${NC}"

    # Test endpoints
    echo ""
    echo "${YELLOW}[4/4] Testing API Endpoints...${NC}"
    echo "----------------------------------------"

    # Health check
    echo "Testing /health..."
    curl -s http://localhost:8000/health | python3.10 -m json.tool

    echo ""
    echo "Testing /info..."
    curl -s http://localhost:8000/info | python3.10 -m json.tool

    echo ""
    echo "Testing /predict (single)..."
    python3.10 examples/test_predict_single.py

    echo ""
    echo "${GREEN}✓ API endpoints working!${NC}"
else
    echo "${YELLOW}⚠ API server not running (this is OK)${NC}"
    echo "To test API endpoints:"
    echo "  1. Start server: make serve"
    echo "  2. In another terminal: make test-api"
fi
echo ""

# Summary
echo "=========================================="
echo "${GREEN}All Serving Tests Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  • View coverage: open htmlcov/index.html"
echo "  • Start server: make serve"
echo "  • Test API: make test-api"
echo "  • Docker test: make docker-build && make docker-run"
echo ""
