#!/bin/bash
# Build Docker image with version tagging

set -e  # Exit on error

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

# Image name (local)
IMAGE_NAME="ml-service"

# DockerHub image name (optional, set via environment variable)
DOCKERHUB_IMAGE="${DOCKERHUB_IMAGE:-artemiop/mlops-news-predictor}"
TAG_DOCKERHUB="${TAG_DOCKERHUB:-true}"  # Set to "false" to disable DockerHub tagging

# Get version from git (tag or commit sha)
if git rev-parse --git-dir > /dev/null 2>&1; then
    GIT_SHA=$(git rev-parse --short HEAD)
    GIT_TAG=$(git describe --tags --exact-match 2>/dev/null || echo "")
else
    GIT_SHA="unknown"
    GIT_TAG=""
fi

# Build arguments
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "================================================"
echo "Building Docker image: $IMAGE_NAME"
echo "Git SHA: $GIT_SHA"
echo "Git Tag: ${GIT_TAG:-none}"
echo "Build Date: $BUILD_DATE"
echo "================================================"

# Build the image
docker build \
    --build-arg BUILD_DATE="$BUILD_DATE" \
    --build-arg GIT_SHA="$GIT_SHA" \
    -t "${IMAGE_NAME}:latest" \
    -t "${IMAGE_NAME}:${GIT_SHA}" \
    .

# If there's a git tag, also tag with that
if [ -n "$GIT_TAG" ]; then
    docker tag "${IMAGE_NAME}:latest" "${IMAGE_NAME}:${GIT_TAG}"
    echo "Tagged image: ${IMAGE_NAME}:${GIT_TAG}"
fi

# Tag for DockerHub (if enabled)
if [ "$TAG_DOCKERHUB" = "true" ]; then
    echo ""
    echo "Tagging for DockerHub: $DOCKERHUB_IMAGE"

    # Always tag latest
    docker tag "${IMAGE_NAME}:latest" "${DOCKERHUB_IMAGE}:latest"
    echo "  ✓ ${DOCKERHUB_IMAGE}:latest"

    # Tag with git SHA
    docker tag "${IMAGE_NAME}:latest" "${DOCKERHUB_IMAGE}:${GIT_SHA}"
    echo "  ✓ ${DOCKERHUB_IMAGE}:${GIT_SHA}"

    # If git tag exists, also tag with that
    if [ -n "$GIT_TAG" ]; then
        docker tag "${IMAGE_NAME}:latest" "${DOCKERHUB_IMAGE}:${GIT_TAG}"
        echo "  ✓ ${DOCKERHUB_IMAGE}:${GIT_TAG}"
    fi
fi

echo ""
echo "================================================"
echo "Build completed successfully!"
echo "================================================"
echo "Local tags:"
docker images "${IMAGE_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

if [ "$TAG_DOCKERHUB" = "true" ]; then
    echo ""
    echo "DockerHub tags:"
    docker images "${DOCKERHUB_IMAGE}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
    echo ""
    echo "To push to DockerHub:"
    echo "  make docker-push"
    echo "  # Or manually:"
    echo "  docker push ${DOCKERHUB_IMAGE}:latest"
    if [ -n "$GIT_TAG" ]; then
        echo "  docker push ${DOCKERHUB_IMAGE}:${GIT_TAG}"
    fi
fi

echo ""
echo "To run the container:"
echo "  make docker-run"
echo "  # Or manually:"
echo "  docker run -p 8000:8000 -v \$(pwd)/models:/app/models ${IMAGE_NAME}:latest"
echo ""
echo "Or use docker-compose:"
echo "  make docker-up"
