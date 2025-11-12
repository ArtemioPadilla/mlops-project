#!/bin/bash
# Start FastAPI server with correct Python version

set -e

echo "=========================================="
echo "Starting FastAPI Server"
echo "=========================================="
echo ""

# Check if port 8000 is already in use
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Port 8000 is already in use"
    echo "Killing existing process..."
    lsof -ti :8000 | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# Check Python version
PYTHON_CMD="python3.10"

if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "❌ $PYTHON_CMD not found!"
    echo "Please install Python 3.10 or update PYTHON_CMD in this script"
    exit 1
fi

echo "Using: $($PYTHON_CMD --version)"
echo ""

# Check if python-multipart is installed
if ! $PYTHON_CMD -c "import multipart" 2>/dev/null; then
    echo "⚠️  python-multipart not installed"
    echo "Installing dependencies..."
    $PYTHON_CMD -m pip install python-multipart uvicorn[standard]
fi

# Start server
echo "Starting server on http://0.0.0.0:8000"
echo "Press CTRL+C to stop"
echo ""
echo "Available endpoints:"
echo "  • http://localhost:8000          - API info"
echo "  • http://localhost:8000/docs     - Swagger UI"
echo "  • http://localhost:8000/redoc    - ReDoc"
echo "  • http://localhost:8000/health   - Health check"
echo ""

$PYTHON_CMD -m uvicorn mlops_online_news_popularity.serving.app:app \
    --reload \
    --host 0.0.0.0 \
    --port 8000
