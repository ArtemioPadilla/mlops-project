# Multi-stage Dockerfile for ML Model Serving
# Stage 1: Builder - Install dependencies
FROM python:3.10-slim AS builder

# Set working directory
WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime - Copy only what's needed
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies (curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user for security
RUN useradd -m -u 1000 mluser && \
    chown -R mluser:mluser /app

# Copy application code
COPY --chown=mluser:mluser mlops_online_news_popularity /app/mlops_online_news_popularity
COPY --chown=mluser:mluser pyproject.toml /app/
COPY --chown=mluser:mluser README.md /app/

# Create directories for models and mlflow artifacts (will be mounted as volumes)
RUN mkdir -p /app/models /app/mlflow_artifacts && \
    chown -R mluser:mluser /app/models /app/mlflow_artifacts

# Install the package in editable mode
RUN pip install -e .

# Switch to non-root user
USER mluser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Environment variables (can be overridden)
ENV MODEL_NAME=RandomForestBase \
    MODEL_LOAD_STRATEGY=local \
    MODEL_PATH=/app/models/randomforestbase_best_20251102_165526.pkl \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    LOG_LEVEL=INFO

# Run the application
CMD ["uvicorn", "mlops_online_news_popularity.serving.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--log-level", "info"]
