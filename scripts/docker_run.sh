#!/bin/bash
# Run Docker container with proper volume mounts

set -e  # Exit on error

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

# Configuration
IMAGE_NAME="ml-service:latest"
CONTAINER_NAME="online-news-predictor"
PORT="8000"

# Environment variables (can be overridden)
MODEL_NAME="${MODEL_NAME:-RandomForestBase}"
MODEL_LOAD_STRATEGY="${MODEL_LOAD_STRATEGY:-local}"
MODEL_PATH="${MODEL_PATH:-/app/models/randomforestbase_best_20251102_165526.pkl}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

echo "================================================"
echo "Running Docker container: $CONTAINER_NAME"
echo "Image: $IMAGE_NAME"
echo "Port: $PORT"
echo "Model: $MODEL_NAME"
echo "Load Strategy: $MODEL_LOAD_STRATEGY"
echo "================================================"

# Check if container is already running
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container $CONTAINER_NAME already exists. Stopping and removing..."
    docker stop "$CONTAINER_NAME" || true
    docker rm "$CONTAINER_NAME" || true
fi

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/logs"

# Run the container
docker run -d \
    --name "$CONTAINER_NAME" \
    -p "${PORT}:8000" \
    -v "$PROJECT_ROOT/models:/app/models:ro" \
    -v "$PROJECT_ROOT/mlflow_artifacts:/app/mlflow_artifacts:ro" \
    -v "$PROJECT_ROOT/logs:/app/logs" \
    -e MODEL_NAME="$MODEL_NAME" \
    -e MODEL_LOAD_STRATEGY="$MODEL_LOAD_STRATEGY" \
    -e MODEL_PATH="$MODEL_PATH" \
    -e LOG_LEVEL="$LOG_LEVEL" \
    --restart unless-stopped \
    "$IMAGE_NAME"

echo ""
echo "Container started successfully!"
echo ""
echo "Waiting for service to be healthy..."
sleep 5

# Check health
for i in {1..30}; do
    if curl -sf http://localhost:${PORT}/health > /dev/null; then
        echo "Service is healthy!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "Warning: Service did not become healthy within 30 seconds"
        echo "Check logs with: docker logs $CONTAINER_NAME"
    fi
    sleep 1
done

echo ""
echo "================================================"
echo "Service is running!"
echo "================================================"
echo "API Documentation: http://localhost:${PORT}/docs"
echo "Health Check:      http://localhost:${PORT}/health"
echo "Model Info:        http://localhost:${PORT}/info"
echo ""
echo "View logs:         docker logs -f $CONTAINER_NAME"
echo "Stop container:    docker stop $CONTAINER_NAME"
echo "Remove container:  docker rm $CONTAINER_NAME"
echo "================================================"
