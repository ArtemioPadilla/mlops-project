# Docker Hub Publishing Guide

Complete guide for publishing and managing Docker images on Docker Hub for the MLOps News Popularity Predictor project.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Initial Setup](#initial-setup)
- [Building and Tagging Images](#building-and-tagging-images)
- [Publishing to Docker Hub](#publishing-to-docker-hub)
- [Pulling and Using Images](#pulling-and-using-images)
- [Version Management](#version-management)
- [CI/CD Integration](#cicd-integration)
- [Advanced Topics](#advanced-topics)
- [Troubleshooting](#troubleshooting)

---

## Overview

The project's Docker image is hosted on Docker Hub at:

**Repository**: [`artemiop/mlops-news-predictor`](https://hub.docker.com/r/artemiop/mlops-news-predictor)

This image provides a production-ready FastAPI service for serving the trained ML model with:
- Multi-stage build for optimized image size
- Non-root user for security
- Health checks and monitoring
- Volume mounts for models and artifacts
- Configurable via environment variables

---

## Prerequisites

### 1. Docker Hub Account

Create a free account at [hub.docker.com](https://hub.docker.com) if you don't have one.

### 2. Docker Installed

Verify Docker is installed and running:

```bash
docker --version
# Docker version 24.0.0 or higher recommended
```

### 3. Local Image Built

Build the Docker image locally first:

```bash
make docker-build
# Or: docker build -t ml-service:latest .
```

---

## Initial Setup

### 1. Login to Docker Hub

#### Using Username and Password

```bash
docker login
# Enter username: artemiop
# Enter password: ************
```

#### Using Access Token (Recommended for CI/CD)

1. Go to [Docker Hub Settings > Security](https://hub.docker.com/settings/security)
2. Click "New Access Token"
3. Give it a descriptive name (e.g., "mlops-project-ci")
4. Copy the token (you won't see it again!)

```bash
# Login with token
echo "YOUR_TOKEN" | docker login --username artemiop --password-stdin
```

**Best Practice**: Store tokens securely in environment variables or CI/CD secrets, never commit them to git.

### 2. Create Repository on Docker Hub (Optional)

By default, pushing an image creates a public repository automatically. To customize:

1. Go to [Docker Hub](https://hub.docker.com)
2. Click "Create Repository"
3. Name: `mlops-news-predictor`
4. Visibility: Public or Private
5. Description: "MLOps project for predicting online news popularity using FastAPI and scikit-learn"
6. Click "Create"

---

## Building and Tagging Images

### Local Image Tags

The local build creates:

```bash
make docker-build
# Creates: ml-service:latest
# Also creates: ml-service:<git-sha>, ml-service:<git-tag>
```

### Tagging for Docker Hub

Before pushing, tag images with the Docker Hub repository name:

```bash
# Tag latest
docker tag ml-service:latest artemiop/mlops-news-predictor:latest

# Tag with version
docker tag ml-service:latest artemiop/mlops-news-predictor:v1.0.0

# Tag with git commit SHA (for traceability)
GIT_SHA=$(git rev-parse --short HEAD)
docker tag ml-service:latest artemiop/mlops-news-predictor:${GIT_SHA}

# Tag with custom label
docker tag ml-service:latest artemiop/mlops-news-predictor:stable
```

### All-in-One Build and Tag

```bash
# Build with multiple tags directly
docker build \
  -t artemiop/mlops-news-predictor:latest \
  -t artemiop/mlops-news-predictor:v1.0.0 \
  -t artemiop/mlops-news-predictor:$(git rev-parse --short HEAD) \
  .
```

---

## Publishing to Docker Hub

### Push Images

After tagging, push to Docker Hub:

```bash
# Push latest tag
docker push artemiop/mlops-news-predictor:latest

# Push version tag
docker push artemiop/mlops-news-predictor:v1.0.0

# Push all tags for this image
docker push --all-tags artemiop/mlops-news-predictor
```

### Verify Upload

Check your repository at: https://hub.docker.com/r/artemiop/mlops-news-predictor/tags

You should see:
- Tag names (latest, v1.0.0, etc.)
- Image size
- Upload timestamp
- Platform (linux/amd64)

---

## Pulling and Using Images

### Pull from Docker Hub

Anyone can pull and run your public image:

```bash
# Pull latest version
docker pull artemiop/mlops-news-predictor:latest

# Pull specific version
docker pull artemiop/mlops-news-predictor:v1.0.0

# Pull and run in one command
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -e MODEL_NAME=RandomForestBase \
  artemiop/mlops-news-predictor:latest
```

### Run with Custom Configuration

```bash
docker run -d \
  --name news-predictor \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/mlflow_artifacts:/app/mlflow_artifacts:ro \
  -e MODEL_LOAD_STRATEGY=local \
  -e MODEL_PATH=/app/models/randomforestbase_best_20251102_165526.pkl \
  -e LOG_LEVEL=INFO \
  artemiop/mlops-news-predictor:latest
```

### Health Check

```bash
# Wait for container to be healthy
docker ps --filter "name=news-predictor" --format "{{.Status}}"

# Check API health
curl http://localhost:8000/health
```

---

## Version Management

### Semantic Versioning

Follow [Semantic Versioning](https://semver.org/) (MAJOR.MINOR.PATCH):

- `v1.0.0` - Initial release
- `v1.1.0` - New features, backwards compatible
- `v1.0.1` - Bug fixes
- `v2.0.0` - Breaking changes

### Tagging Strategy

| Tag | Purpose | Example |
|-----|---------|---------|
| `latest` | Most recent stable build | `artemiop/mlops-news-predictor:latest` |
| `vX.Y.Z` | Semantic version | `artemiop/mlops-news-predictor:v1.2.3` |
| `<git-sha>` | Specific commit | `artemiop/mlops-news-predictor:a1b2c3d` |
| `stable` | Production-ready | `artemiop/mlops-news-predictor:stable` |
| `dev` | Development builds | `artemiop/mlops-news-predictor:dev` |

### Example: Release Workflow

```bash
# 1. Update version in code (e.g., __version__ = "1.2.0")

# 2. Commit and tag
git add .
git commit -m "Release v1.2.0"
git tag v1.2.0
git push origin main --tags

# 3. Build with multiple tags
docker build \
  -t artemiop/mlops-news-predictor:latest \
  -t artemiop/mlops-news-predictor:v1.2.0 \
  -t artemiop/mlops-news-predictor:stable \
  .

# 4. Push all tags
docker push artemiop/mlops-news-predictor:latest
docker push artemiop/mlops-news-predictor:v1.2.0
docker push artemiop/mlops-news-predictor:stable
```

---

## CI/CD Integration

### GitHub Actions

Create `.github/workflows/docker-publish.yml`:

```yaml
name: Publish Docker Image

on:
  push:
    branches: [main]
    tags: ['v*']
  pull_request:
    branches: [main]

env:
  REGISTRY: docker.io
  IMAGE_NAME: artemiop/mlops-news-predictor

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Docker Hub
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha,prefix={{branch}}-

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Update Docker Hub description
        if: github.event_name != 'pull_request'
        uses: peter-evans/dockerhub-description@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
          repository: artemiop/mlops-news-predictor
          short-description: "MLOps project for predicting online news popularity"
          readme-filepath: ./docs/deployment/DOCKERHUB_README.md
```

### Setup GitHub Secrets

1. Go to repository Settings > Secrets and variables > Actions
2. Add secrets:
   - `DOCKERHUB_USERNAME`: Your Docker Hub username (`artemiop`)
   - `DOCKERHUB_TOKEN`: Your Docker Hub access token

### Manual Makefile Approach

Add to `Makefile`:

```makefile
DOCKER_USERNAME ?= artemiop
DOCKER_IMAGE ?= mlops-news-predictor
VERSION ?= $(shell git describe --tags --always --dirty)

.PHONY: docker-tag
docker-tag:
	docker tag ml-service:latest $(DOCKER_USERNAME)/$(DOCKER_IMAGE):latest
	docker tag ml-service:latest $(DOCKER_USERNAME)/$(DOCKER_IMAGE):$(VERSION)

.PHONY: docker-push
docker-push: docker-tag
	docker push $(DOCKER_USERNAME)/$(DOCKER_IMAGE):latest
	docker push $(DOCKER_USERNAME)/$(DOCKER_IMAGE):$(VERSION)

.PHONY: docker-release
docker-release: docker-build docker-push
	@echo "Released $(DOCKER_USERNAME)/$(DOCKER_IMAGE):$(VERSION)"
```

Usage:

```bash
# Build, tag, and push
make docker-release

# Or step by step
make docker-build
make docker-tag
make docker-push
```

---

## Advanced Topics

### Multi-Architecture Builds

Build for multiple platforms (AMD64 + ARM64):

```bash
# Create buildx builder
docker buildx create --name multiarch --use

# Build and push multi-platform image
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t artemiop/mlops-news-predictor:latest \
  -t artemiop/mlops-news-predictor:v1.0.0 \
  --push \
  .
```

### Image Scanning for Vulnerabilities

```bash
# Scan with Docker Scout (built-in)
docker scout quickview artemiop/mlops-news-predictor:latest
docker scout cves artemiop/mlops-news-predictor:latest

# Scan with Trivy
trivy image artemiop/mlops-news-predictor:latest

# Fail build on HIGH/CRITICAL vulnerabilities
trivy image --exit-code 1 --severity HIGH,CRITICAL artemiop/mlops-news-predictor:latest
```

### Image Size Optimization

Check layer sizes:

```bash
docker history artemiop/mlops-news-predictor:latest --human --format "table {{.Size}}\t{{.CreatedBy}}"
```

Tips to reduce size:
- Use `.dockerignore` (already configured)
- Multi-stage builds (already implemented)
- Minimize layers (combine RUN commands)
- Remove build dependencies in same layer
- Use slim base images (already using `python:3.10-slim`)

### Private Repositories

For private repositories:

```bash
# Pull from private repo (requires authentication)
docker login
docker pull artemiop/mlops-news-predictor:latest

# Or with inline auth
docker login -u artemiop -p $DOCKERHUB_TOKEN
docker pull artemiop/mlops-news-predictor:latest
```

In Kubernetes:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: dockerhub-secret
type: kubernetes.io/dockerconfigjson
data:
  .dockerconfigjson: <base64-encoded-docker-config>
---
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      imagePullSecrets:
      - name: dockerhub-secret
      containers:
      - name: ml-service
        image: artemiop/mlops-news-predictor:latest
```

---

## Troubleshooting

### Authentication Errors

**Problem**: `unauthorized: authentication required`

**Solution**:
```bash
# Re-login
docker logout
docker login

# Check credentials
cat ~/.docker/config.json
```

### Push Denied

**Problem**: `denied: requested access to the resource is denied`

**Solutions**:
- Verify you're logged in: `docker login`
- Check repository name matches: `artemiop/mlops-news-predictor`
- Ensure you have write permissions (own the repository)

### Rate Limiting

**Problem**: `toomanyrequests: You have reached your pull rate limit`

**Solutions**:
- Authenticate (increases limit from 100 to 200 pulls/6h)
- Upgrade to Docker Pro/Team
- Use caching strategies
- Use a private registry mirror

### Large Upload Times

**Problem**: Push takes too long

**Solutions**:
```bash
# Use layer caching
docker build --cache-from artemiop/mlops-news-predictor:latest -t artemiop/mlops-news-predictor:latest .

# Compress before pushing (automatic, but verify)
docker push artemiop/mlops-news-predictor:latest --compress

# Check network speed
speedtest-cli
```

### Image Not Found After Push

**Problem**: Image pushed successfully but shows as "not found"

**Solutions**:
- Wait 1-2 minutes for Docker Hub to index
- Refresh browser cache
- Verify tag: `docker search artemiop/mlops-news-predictor`
- Check repository visibility (public vs private)

### Tag Conflicts

**Problem**: Tag already exists with different content

**Solutions**:
```bash
# Force overwrite (not recommended for production)
docker push artemiop/mlops-news-predictor:latest

# Use new tag instead
docker tag ml-service:latest artemiop/mlops-news-predictor:v1.0.1
docker push artemiop/mlops-news-predictor:v1.0.1
```

---

## Best Practices

### 1. Version Everything

Always tag with versions, never rely only on `latest`:

```bash
docker push artemiop/mlops-news-predictor:v1.2.3  # Good
docker push artemiop/mlops-news-predictor:latest  # Also push latest
```

### 2. Use Immutable Tags for Production

Never overwrite production tags. Create new versions instead:

```bash
# Bad: Overwriting production tag
docker push artemiop/mlops-news-predictor:stable  # Danger!

# Good: New version
docker push artemiop/mlops-news-predictor:v1.2.4
# Then update stable to point to v1.2.4
```

### 3. Keep Images Small

Monitor image size:

```bash
docker images artemiop/mlops-news-predictor --format "{{.Tag}}\t{{.Size}}"
```

Target: < 500MB for ML services (current: ~300MB without model, ~2.7GB with model)

### 4. Security

- Scan for vulnerabilities before production
- Use specific base image versions (not `latest`)
- Run as non-root user (already implemented)
- Use read-only volumes where possible (already implemented)

### 5. Documentation

Update Docker Hub repository README with:
- Quick start instructions
- Available tags and their meanings
- Environment variables
- Volume mounts
- Health check endpoints

---

## Related Documentation

- [Docker Deployment Overview](../serving/deployment.md)
- [Docker Success Report](../../DOCKER_SUCCESS.md)
- [API Documentation](../API_DOCUMENTATION.md)
- [GitHub Actions Workflows](../workflows/)

---

## Quick Reference

### Essential Commands

```bash
# Login
docker login

# Build
docker build -t artemiop/mlops-news-predictor:latest .

# Tag
docker tag ml-service:latest artemiop/mlops-news-predictor:v1.0.0

# Push
docker push artemiop/mlops-news-predictor:latest

# Pull
docker pull artemiop/mlops-news-predictor:latest

# Run
docker run -p 8000:8000 -v $(pwd)/models:/app/models:ro artemiop/mlops-news-predictor:latest
```

### Makefile Shortcuts

```bash
make docker-build    # Build image
make docker-tag      # Tag for DockerHub (if implemented)
make docker-push     # Push to DockerHub (if implemented)
make docker-release  # Build + Tag + Push (if implemented)
```

---

**Last Updated**: November 2024
**Maintainer**: Artemio Padilla
**Docker Hub**: https://hub.docker.com/r/artemiop/mlops-news-predictor
