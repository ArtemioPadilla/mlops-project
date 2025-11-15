# Model Serving & Deployment

## Overview

The Online News Popularity prediction model is served via a production-ready **FastAPI** application that provides RESTful endpoints for both online (single) and batch predictions. The service is fully containerized with Docker and designed for easy deployment to various platforms.

## Architecture

### System Context: High-Level Architecture

This diagram shows the high-level view of the News Popularity Prediction System and its interactions with users and external systems.

```mermaid
flowchart TB
    subgraph actors["👥 Users & Stakeholders"]
        DS["<b>Data Scientist</b><br/>Trains ML models<br/>Deploys to production<br/>Validates predictions"]
        AC["<b>API Consumer</b><br/>Web apps, scripts, notebooks<br/>Sends prediction requests<br/>Receives popularity scores"]
        DevOps["<b>DevOps Engineer</b><br/>Manages infrastructure<br/>Monitors health<br/>Handles deployments"]
    end

    subgraph core["🚀 News Popularity Prediction Service"]
        Service["<b>FastAPI ML Serving Platform</b><br/><br/>• REST API endpoints<br/>• Input validation (59 features)<br/>• ModelHandler inference pipeline<br/>• Health monitoring<br/>• Automatic documentation"]
    end

    subgraph external["🔗 External Systems"]
        MLF["<b>MLflow Registry</b><br/>Model versioning<br/>Experiment tracking<br/>Artifact storage"]
        MON["<b>Monitoring System</b><br/>Metrics collection<br/>Log aggregation<br/>Dashboards & alerts<br/>(Prometheus/Grafana)"]
    end

    DS -->|"Train & register<br/>models"| MLF
    DS -->|"Deploy models<br/>Validate predictions<br/>[REST API]"| Service

    AC -->|"Send article features<br/>Receive share predictions<br/>[HTTPS/JSON]"| Service

    Service -->|"Load trained models<br/>& metadata<br/>[MLflow Client]"| MLF
    Service -->|"Export logs<br/>& metrics<br/>[HTTP]"| MON

    DevOps -->|"Health checks<br/>Manage deployments<br/>[Docker/K8s]"| Service
    DevOps -->|"View dashboards<br/>Monitor alerts<br/>[Web UI]"| MON

    style actors fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000
    style core fill:#e8f5e9,stroke:#388e3c,stroke-width:3px,color:#000
    style external fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000
    style Service fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px,color:#000
```

**Key Interactions:**

- **Data Scientists** train models using the MLOps pipeline and register them in MLflow, then validate predictions through the API
- **API Consumers** (web apps, scripts, notebooks) send article features and receive popularity predictions
- **DevOps Engineers** deploy containers, monitor health endpoints, and manage infrastructure
- **MLflow Registry** provides trained models and metadata for inference
- **Monitoring System** collects operational metrics and logs (Prometheus/Grafana/CloudWatch)

---

### Container Architecture: Internal Structure

This diagram zooms into the Prediction Service to show its internal containers (applications and data stores).

```mermaid
flowchart LR
    Client["🔌 <b>API Client</b><br/>External applications<br/>Web apps, scripts"]
    DevOps["🔧 <b>DevOps</b><br/>Infrastructure<br/>management"]

    subgraph boundary["📦 News Popularity Prediction Service"]
        direction TB

        API["<b>FastAPI Application</b><br/><i>Python, FastAPI, Uvicorn</i><br/><br/>• Port 8000<br/>• 6 REST endpoints<br/>• Auto documentation<br/>• CORS middleware"]

        subgraph processing["⚙️ Request Processing Layer"]
            direction LR
            Val["<b>Pydantic Schemas</b><br/><i>Python, Pydantic</i><br/><br/>Validates 59<br/>article features"]

            Handler["<b>ModelHandler</b><br/><i>Python, sklearn</i><br/><br/>• preprocess()<br/>• inference()<br/>• postprocess()<br/>• handle()"]
        end

        subgraph storage["💾 Storage & Configuration"]
            direction TB
            LocalModels["<b>Local Model Storage</b><br/><i>File System</i><br/><br/>Pickle files (.pkl)<br/>Read-only volume<br/>Mounted at /app/models"]

            Config["<b>Configuration</b><br/><i>.env, YAML</i><br/><br/>Environment vars<br/>API settings<br/>Model paths"]
        end

        API --> Val
        API --> Handler
        Val -.->|"Schemas used<br/>for DataFrame"| Handler
    end

    subgraph ext["🔗 External Resources"]
        direction TB
        MLF["<b>MLflow Registry</b><br/>Model versioning<br/>Artifacts & metadata"]
        MON["<b>Monitoring</b><br/>Logs & metrics<br/>Dashboards"]
    end

    Client -->|"Prediction requests<br/>[HTTPS/JSON]"| API

    Handler -->|"Load on startup<br/>[joblib.load()]"| LocalModels
    Handler -->|"Optional load<br/>[MLflow Client]"| MLF

    API -->|"Read settings<br/>[os.getenv()]"| Config
    Handler -->|"Read model path<br/>& strategy"| Config

    API -->|"Export logs<br/>& metrics<br/>[Loguru, HTTP]"| MON

    DevOps -->|"Health checks<br/>[GET /health]"| API
    DevOps -->|"Mount volume<br/>[Docker]"| LocalModels

    style boundary fill:#e8f5e9,stroke:#388e3c,stroke-width:3px,color:#000
    style processing fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px,color:#000
    style storage fill:#f1f8e9,stroke:#558b2f,stroke-width:2px,color:#000
    style ext fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000
    style API fill:#a5d6a7,stroke:#1b5e20,stroke-width:2px,color:#000
```

**Key Containers:**

- **FastAPI Application**: Web server exposing 6 REST endpoints (`/health`, `/info`, `/predict`, `/predict/batch`, `/predict/batch/csv`, `/docs`)
- **ModelHandler**: Core inference engine implementing the prediction pipeline (preprocess → inference → postprocess)
- **Pydantic Schemas**: Data validation layer ensuring all 59 features are present and valid before inference
- **Local Model Storage**: File system volume containing trained sklearn Pipeline models (`.pkl` files)
- **Configuration**: Environment-based settings for model loading strategy, API host/port, and log levels

**Data Flow:**

1. API consumer sends prediction request to FastAPI
2. FastAPI validates input against Pydantic schemas
3. FastAPI passes validated data to ModelHandler
4. ModelHandler loads model from local storage or MLflow (on first request)
5. ModelHandler runs prediction pipeline and returns results
6. FastAPI formats response and returns to consumer

---

### Detailed Flows

The following diagrams provide deeper technical details about request processing and state management.

#### Request Flow

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant ModelHandler
    participant Validation
    participant Model

    Client->>FastAPI: POST /predict
    FastAPI->>ModelHandler: handle(input_data)

    ModelHandler->>Validation: preprocess()
    Validation->>Validation: Validate 59 features
    Validation->>Validation: Convert to DataFrame

    alt Validation Failed
        Validation-->>Client: 422 Validation Error
    end

    Validation->>Model: inference()
    Model->>Model: Scale features
    Model->>Model: Run RandomForest
    Model-->>ModelHandler: predictions (log scale)

    ModelHandler->>ModelHandler: postprocess()
    ModelHandler->>ModelHandler: Apply expm1 (inverse log)
    ModelHandler->>ModelHandler: Round to integers

    ModelHandler-->>FastAPI: {predicted_shares, log_prediction}
    FastAPI-->>Client: 200 OK + JSON response
```

#### ModelHandler State Machine

This state diagram shows the complete lifecycle of the ModelHandler from initialization through prediction.

```mermaid
stateDiagram-v2
    [*] --> Uninitialized
    Uninitialized --> Loading: initialize()
    Loading --> Ready: Model Loaded
    Loading --> Error: Load Failed

    Ready --> Preprocessing: handle() called
    Preprocessing --> Validating: Pydantic validation
    Validating --> Preprocessing: Valid
    Validating --> Error: Invalid (422)

    Preprocessing --> Inference: DataFrame ready
    Inference --> Pipeline: sklearn.predict()
    Pipeline --> Postprocessing: predictions (log scale)

    Postprocessing --> Transform: expm1()
    Transform --> Format: Round to int
    Format --> [*]: Return predictions

    Error --> [*]: HTTP Error

    note right of Ready
        Model cached in memory
        Ready for predictions
    end note

    note right of Pipeline
        1. StandardScaler
        2. RandomForest
        Returns log(shares)
    end note

    note right of Transform
        Inverse log transform:
        shares = exp(pred) - 1
    end note
```

#### ModelHandler Methods

This flowchart shows the internal methods of ModelHandler and how they interact.

```mermaid
flowchart LR
    subgraph "initialize()"
        INIT_START[Start]
        INIT_CHECK{Load Strategy?}
        INIT_LOCAL[Load from<br/>Pickle File]
        INIT_MLF[Load from<br/>MLflow Registry]
        INIT_END[Model Ready]

        INIT_START --> INIT_CHECK
        INIT_CHECK -->|local| INIT_LOCAL
        INIT_CHECK -->|mlflow| INIT_MLF
        INIT_LOCAL --> INIT_END
        INIT_MLF --> INIT_END
    end

    subgraph "handle()"
        HANDLE_START[Input Data]
        HANDLE_PRE[preprocess]
        HANDLE_INF[inference]
        HANDLE_POST[postprocess]
        HANDLE_END[Output]

        HANDLE_START --> HANDLE_PRE
        HANDLE_PRE --> HANDLE_INF
        HANDLE_INF --> HANDLE_POST
        HANDLE_POST --> HANDLE_END
    end

    INIT_END -.->|Ready| HANDLE_START

    style INIT_END fill:#90EE90
    style HANDLE_END fill:#90EE90
```

---

## Key Features

- **Multiple Endpoints**: Health check, model info, single prediction, batch prediction (JSON & CSV)
- **Input Validation**: Pydantic-based schema validation for all 59 features
- **Flexible Model Loading**: Load from local pickle files or MLflow registry
- **Automatic Documentation**: Interactive Swagger UI and ReDoc
- **Error Handling**: Comprehensive error messages with proper HTTP status codes
- **Logging**: Performance timing and debugging logs via loguru
- **CORS Support**: Configurable CORS middleware for web integration
- **Health Monitoring**: Built-in health check endpoint for container orchestration

## Quick Start

### Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the server (with auto-reload)
make serve

# 3. Access the API
open http://localhost:8000/docs
```

### Docker Deployment

```bash
# 1. Build the image
make docker-build

# 2. Run the container
make docker-up

# 3. View logs
make docker-logs
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and links |
| `/health` | GET | Health check status |
| `/info` | GET | Model metadata and features |
| `/predict` | POST | Single article prediction |
| `/predict/batch` | POST | Batch prediction (JSON) |
| `/predict/batch/csv` | POST | Batch prediction (CSV upload) |
| `/docs` | GET | Swagger UI documentation |
| `/redoc` | GET | ReDoc documentation |

## Components

### ModelHandler

The `ModelHandler` class implements a pattern similar to AWS SageMaker handlers and orchestrates the complete ML prediction pipeline:

- **`initialize()`**: Loads model from MLflow registry or local pickle file on startup
- **`preprocess()`**: Validates input features and converts to DataFrame format
- **`inference()`**: Runs prediction through sklearn Pipeline (scaling + model)
- **`postprocess()`**: Applies inverse log transform and formats output as integers
- **`handle()`**: Orchestrates the complete pipeline (preprocess → inference → postprocess)

See the [Detailed Flows](#detailed-flows) section for visual diagrams of the ModelHandler state machine and methods.

### FastAPI Application

The FastAPI app (`app.py`) provides:

- RESTful endpoints with OpenAPI documentation
- Request/response validation via Pydantic
- Error handling with detailed messages
- CORS middleware for cross-origin requests
- Startup/shutdown event handlers

### Configuration

Environment-based configuration via `.env` file:

```bash
MODEL_NAME=RandomForestBase
MODEL_LOAD_STRATEGY=local  # or mlflow
MODEL_PATH=models/randomforestbase_best_20251102_165526.pkl
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

## Performance

### Response Times (Approximate)

- **Health check**: <10ms
- **Model info**: <10ms
- **Single prediction**: 50-100ms
- **Batch (100 articles)**: 200-500ms
- **CSV upload (100 articles)**: 300-600ms

### Optimization Tips

1. Use batch endpoints for multiple predictions
2. Keep batch sizes under 500 for optimal performance
3. CSV format is slightly faster than JSON for large batches
4. Reuse HTTP connections when making multiple requests

## Testing

The serving module has comprehensive test coverage:

```bash
# Run all serving tests
make test-serving

# Run with coverage report
make test-coverage

# Run integration tests only
make test-integration
```

**Test Coverage**: 80%+ with 83 tests covering:
- Pydantic schema validation
- ModelHandler pipeline
- All API endpoints
- Error handling
- CORS and documentation

## Next Steps

- [Getting Started](getting-started.md) - Step-by-step setup guide
- [API Reference](api-reference.md) - Complete endpoint documentation
- [Deployment Guide](deployment.md) - Production deployment options
- [Testing Guide](testing.md) - How to test the API
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
