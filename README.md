# RLM Document Retrieval System

A Recursive Language Model (RLM) based document retrieval system capable of processing documents up to **10M+ tokens** with intelligent recursive analysis.

## Overview

This system implements the RLM architecture from the [RLM paper](https://arxiv.org/pdf/2512.24601), enabling deep document analysis through recursive code generation and execution. Instead of traditional RAG approaches that lose context in long documents, RLM dynamically generates Python code to explore and analyze documents at any depth.

### Key Features

- **10M+ Token Support**: Chunked context loading with lazy evaluation for massive documents
- **100+ LLM Providers**: Unified interface via LiteLLM (OpenAI, Anthropic, Azure, Google, local models)
- **Secure Code Execution**: Docker-based sandboxing for untrusted code
- **Real-time Streaming**: Live trajectory updates and result streaming
- **Intelligent Query Routing**: Automatic selection between Direct LLM, RAG, RLM, and Hybrid strategies
- **Cost Tracking**: Per-query cost estimation and tracking with detailed reports
- **Interactive Visualization**: React-based UI for exploring execution trajectories

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
│  (Web UI + CLI + API Endpoints)                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  Orchestration Layer                         │
│  • Query Router    • Session Manager   • Cost Tracker       │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                    RLM Core Engine                           │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Root LLM    │  │ REPL Manager │  │ Sub-LLM Pool │       │
│  │ Controller  │◄─┤ (Sandboxed)  │◄─┤ (Async)      │       │
│  └─────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              Document Processing Layer                       │
│  • Ingestion Pipeline  • Chunking Engine                    │
│  • Metadata Extractor  • Format Converters                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

Before you begin, ensure you have the following installed:

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python** | 3.11 or higher | Required |
| **Node.js** | 20+ | Required for frontend |
| **npm** | 10+ | Comes with Node.js |
| **Git** | Any recent version | For cloning |
| **Docker Desktop** | Latest | Optional - for secure sandboxing |
| **Qdrant** | Latest | Optional - for RAG/Hybrid mode |

### Checking Your Versions

```powershell
# Windows PowerShell
python --version    # Should be 3.11+
node --version      # Should be 20+
npm --version       # Should be 10+
```

---

## Step-by-Step Setup Guide

### Step 1: Install UV Package Manager

UV is a fast Python package manager (10x faster than pip). Install it first:

```powershell
# Windows (PowerShell or Command Prompt)
pip install uv

# Linux/Mac
pip install uv
```

Verify installation:
```powershell
uv --version
```

### Step 2: Navigate to Project Directory

```powershell
cd d:\projects\RLM_demo\rlm-document-retrieval
```

### Step 3: Create Virtual Environment with UV

```powershell
# Creates .venv directory with Python 3.11
uv venv .venv --python 3.11
```

### Step 4: Activate Virtual Environment

```powershell
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# Linux/Mac
source .venv/bin/activate
```

> **Note**: If you get an execution policy error in PowerShell, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### Step 5: Install Backend Dependencies with UV

```powershell
# Install all dependencies including dev and docker extras
uv pip install -e ".[dev,docker]"
```

This installs:
- Core RLM dependencies (FastAPI, LiteLLM, etc.)
- Development tools (pytest, ruff, mypy)
- Docker SDK for Python

### Step 6: Configure Environment Variables

```powershell
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

Edit the `.env` file and add your API keys:

```ini
# Required: At least one LLM provider API key
OPENAI_API_KEY=your-openai-api-key-here

# Optional: Additional providers
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# LLM Configuration
RLM_LITELLM_PROVIDER=openai
RLM_DEFAULT_MODEL=gpt-4o-mini

# Recursion Limits
RLM_MAX_RECURSION_DEPTH=3
RLM_MAX_SUB_LLM_CALLS=100

# Timeouts (seconds)
RLM_CODE_EXECUTION_TIMEOUT=30
RLM_LLM_TIMEOUT=60

# Context Handling (for 10M+ tokens)
RLM_CONTEXT_CHUNK_SIZE=100000
RLM_MAX_CONTEXT_CHUNKS_IN_MEMORY=10

# Logging
RLM_LOG_LEVEL=INFO
RLM_LOG_DIR=./logs
RLM_ENABLE_TRAJECTORY_LOGGING=true
```

### Step 7: Create Required Directories

```powershell
# Windows
mkdir logs

# Linux/Mac
mkdir -p logs
```

### Step 8: Install Frontend Dependencies

```powershell
cd frontend
npm install
cd ..
```

### Step 9: Start the Backend Server

Open a terminal and run:

```powershell
# From project root (rlm-document-retrieval/)
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
```

### Step 10: Start the Frontend (New Terminal)

Open a **new terminal**, activate the environment if needed, and run:

```powershell
cd frontend
npm run dev
```

You should see:
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:3001/
  ➜  Network: http://x.x.x.x:3001/
```

### Step 11: Verify Installation

1. **Backend API Documentation**: Open http://localhost:8000/docs
   - You should see the FastAPI Swagger UI with all endpoints

2. **Frontend UI**: Open http://localhost:3001
   - You should see the Trajectory Visualizer interface

3. **Run a quick test**:
   ```powershell
   pytest backend/tests/test_core.py -v -k "test_chunked"
   ```

---

## Running Tests

### All Tests

```powershell
# Run all tests with verbose output
pytest backend/tests/ -v

# Run with coverage report
pytest backend/tests/ --cov=backend/rlm --cov-report=html
# Open htmlcov/index.html to view coverage report
```

### Component-Specific Tests

```powershell
# RLM Core Engine tests
pytest backend/tests/test_core.py -v

# Docker Sandbox tests (requires Docker)
RUN_DOCKER_TESTS=1 pytest backend/tests/test_docker_sandbox.py -v

# LLM Client tests
pytest backend/tests/test_llm_client.py -v

# Sub-LLM Manager tests
pytest backend/tests/test_sub_llm_manager.py -v

# Document Ingestion tests
pytest backend/tests/test_ingestion.py -v

# Query Router tests
pytest backend/tests/test_routing.py -v
```

### Code Quality

```powershell
# Check for linting issues
ruff check backend/

# Auto-format code
ruff format backend/

# Type checking
mypy backend/rlm
```

---

## Optional: Docker Sandbox Setup

The Docker sandbox provides secure, isolated code execution. This is recommended for production use or when running untrusted code.

### Step 1: Install Docker Desktop

1. Download Docker Desktop for Windows from: https://www.docker.com/products/docker-desktop/
2. Run the installer and follow the prompts
3. **Important**: Enable WSL 2 backend when prompted (recommended for Windows)
4. Restart your computer if required

### Step 2: Verify Docker Installation

```powershell
docker --version
# Docker version 24.x.x or higher

docker run hello-world
# Should print "Hello from Docker!"
```

### Step 3: Pull the Python Image

```powershell
docker pull python:3.11-slim
```

### Step 4: Configure Docker Sandbox

Add to your `.env` file:

```ini
# Enable Docker sandbox
RLM_SANDBOX_TYPE=docker

# Docker configuration
DOCKER_IMAGE=python:3.11-slim
DOCKER_MEMORY_LIMIT=512m
DOCKER_CPU_LIMIT=1.0
DOCKER_SECURITY_PROFILE=strict
DOCKER_NETWORK_ENABLED=false
DOCKER_AUTO_CLEANUP=true
```

### Step 5: Test Docker Sandbox

```powershell
# Run Docker-specific tests
$env:RUN_DOCKER_TESTS=1; pytest backend/tests/test_docker_sandbox.py -v

# Or on Linux/Mac
RUN_DOCKER_TESTS=1 pytest backend/tests/test_docker_sandbox.py -v
```

### Security Profiles

| Profile | Use Case | Features |
|---------|----------|----------|
| `strict` | Production | Read-only FS, no network, 50 PID limit |
| `standard` | Development | Read-only FS, bridge network, 100 PID limit |
| `development` | Testing | Writable FS, network enabled, 500 PID limit |

---

## Optional: Qdrant Vector Database Setup

Qdrant enables RAG (Retrieval-Augmented Generation) and Hybrid query modes for efficient document retrieval.

### Step 1: Install Qdrant via Docker

```powershell
# Pull the Qdrant image
docker pull qdrant/qdrant

# Run Qdrant with persistent storage
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

### Step 2: Verify Qdrant is Running

```powershell
# Check container status
docker ps | findstr qdrant

# Test the API
curl http://localhost:6333/collections
# Or open http://localhost:6333/dashboard in browser
```

### Step 3: Configure Qdrant in RLM

Add to your `.env` file:

```ini
# Qdrant Configuration
ROUTING_QDRANT_HOST=localhost
ROUTING_QDRANT_PORT=6333
ROUTING_QDRANT_COLLECTION=rlm_chunks

# Enable embeddings during document ingestion
ROUTING_ENABLE_EMBEDDINGS=true

# Query routing thresholds
ROUTING_DIRECT_LLM_MAX_CONTEXT=10000
ROUTING_RAG_MAX_CONTEXT=500000
```

### Step 4: Using RAG/Hybrid Mode

When Qdrant is configured, the Query Router automatically selects the optimal strategy:

| Query Type | Context Size | Strategy |
|------------|--------------|----------|
| Simple question | < 10K chars | Direct LLM |
| Single-hop retrieval | < 500K chars | RAG |
| Multi-hop reasoning | Any | RLM |
| Large docs + complex | > 500K chars | Hybrid (RAG + RLM) |

### Managing Qdrant

```powershell
# Stop Qdrant
docker stop qdrant

# Start Qdrant
docker start qdrant

# View logs
docker logs qdrant

# Remove container (data persists in volume)
docker rm qdrant

# Remove data volume (caution: deletes all data)
docker volume rm qdrant_storage
```

---

## Configuration Reference

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key (required if using OpenAI) |
| `ANTHROPIC_API_KEY` | - | Anthropic API key (optional) |
| `RLM_LITELLM_PROVIDER` | `openai` | LLM provider (openai, anthropic, azure, etc.) |
| `RLM_DEFAULT_MODEL` | `gpt-4o-mini` | Default LLM model to use |

### Recursion & Limits

| Variable | Default | Description |
|----------|---------|-------------|
| `RLM_MAX_RECURSION_DEPTH` | `3` | Maximum recursion depth |
| `RLM_MAX_SUB_LLM_CALLS` | `100` | Maximum sub-LLM calls per query |
| `RLM_CODE_EXECUTION_TIMEOUT` | `30` | Code execution timeout (seconds) |
| `RLM_LLM_TIMEOUT` | `60` | LLM API timeout (seconds) |

### Context Handling

| Variable | Default | Description |
|----------|---------|-------------|
| `RLM_CONTEXT_CHUNK_SIZE` | `100000` | Tokens per chunk for large docs |
| `RLM_MAX_CONTEXT_CHUNKS_IN_MEMORY` | `10` | Max chunks loaded simultaneously |

### Sandbox Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `RLM_SANDBOX_TYPE` | `auto` | Sandbox type: `auto`, `local`, `docker` |
| `DOCKER_IMAGE` | `python:3.11-slim` | Docker image for sandbox |
| `DOCKER_MEMORY_LIMIT` | `512m` | Container memory limit |
| `DOCKER_CPU_LIMIT` | `1.0` | Container CPU limit |
| `DOCKER_SECURITY_PROFILE` | `standard` | Security profile |

### LLM Client Features

| Variable | Default | Description |
|----------|---------|-------------|
| `LLMCLIENT_ENABLE_RATE_LIMITING` | `true` | Enable rate limiting |
| `LLMCLIENT_ENABLE_COST_TRACKING` | `true` | Enable cost tracking |
| `LLMCLIENT_ENABLE_CIRCUIT_BREAKER` | `true` | Enable circuit breaker |
| `LLMCLIENT_ENABLE_CACHING` | `true` | Enable response caching |

### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `RLM_LOG_LEVEL` | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `RLM_LOG_DIR` | `./logs` | Directory for log files |
| `RLM_ENABLE_TRAJECTORY_LOGGING` | `true` | Enable trajectory JSONL logging |

---

## Project Structure

```
rlm-document-retrieval/
├── backend/
│   └── rlm/
│       ├── core/              # RLM Core Engine
│       │   ├── orchestrator.py
│       │   ├── recursion.py
│       │   └── exceptions.py
│       ├── llm/               # LLM Client & Sub-LLM Manager
│       │   ├── client.py
│       │   ├── enhanced_client.py
│       │   ├── sub_llm_manager.py
│       │   └── ...
│       ├── sandbox/           # Code Execution Sandboxes
│       │   ├── local_repl.py
│       │   ├── docker_repl.py
│       │   └── factory.py
│       ├── documents/         # Document Processing
│       │   ├── ingestion.py
│       │   ├── parsers/
│       │   └── ...
│       ├── routing/           # Query Router
│       │   ├── query_router.py
│       │   ├── analyzer.py
│       │   └── ...
│       ├── trajectory/        # Trajectory Logging
│       │   ├── logger.py
│       │   ├── processor.py
│       │   └── exporter.py
│       ├── api/               # FastAPI Routes
│       │   ├── main.py
│       │   └── routes/
│       └── config/            # Configuration
│           └── settings.py
├── frontend/                  # React + TypeScript UI
│   ├── src/
│   │   ├── components/
│   │   │   └── trajectory/    # Trajectory Visualizer
│   │   ├── stores/
│   │   └── api/
│   ├── package.json
│   └── vite.config.ts
├── tests/                     # Test Suite
├── docs/                      # Documentation
├── examples/                  # Usage Examples
├── scripts/                   # Setup Scripts
├── pyproject.toml            # Python Project Config
├── .env.example              # Environment Template
└── README.md                 # This File
```

---

## Troubleshooting

### PowerShell Execution Policy Error

**Error**: `cannot be loaded because running scripts is disabled on this system`

**Solution**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### UV Not Found

**Error**: `'uv' is not recognized as a command`

**Solution**:
```powershell
# Reinstall UV
pip install --upgrade uv

# Or use pip directly as fallback
pip install -e ".[dev,docker]"
```

### Python Version Error

**Error**: `Python 3.11 not found`

**Solution**:
1. Install Python 3.11+ from https://www.python.org/downloads/
2. Ensure it's added to PATH during installation
3. Restart your terminal

### Docker Not Found

**Error**: `docker: command not found` or `Docker not available`

**Solution**:
1. Install Docker Desktop from https://www.docker.com/products/docker-desktop/
2. Start Docker Desktop
3. Wait for Docker to fully start (check system tray icon)
4. Verify: `docker --version`

### API Key Errors

**Error**: `AuthenticationError` or `Invalid API key`

**Solution**:
1. Check your `.env` file has the correct API key
2. Ensure no extra spaces or quotes around the key
3. Verify the key is valid at your provider's dashboard

### Port Already in Use

**Error**: `Address already in use` or `port 8000 is already allocated`

**Solution**:
```powershell
# Find process using the port (Windows)
netstat -ano | findstr :8000

# Kill the process
taskkill /PID <PID> /F

# Or use a different port
uvicorn backend.rlm.api.main:app --port 8001
```

### Module Not Found

**Error**: `ModuleNotFoundError: No module named 'rlm'`

**Solution**:
1. Ensure virtual environment is activated
2. Reinstall in development mode:
   ```powershell
   uv pip install -e ".[dev,docker]"
   ```

### Frontend Build Errors

**Error**: npm install fails or build errors

**Solution**:
```powershell
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
cd frontend
rmdir /s /q node_modules
npm install
```

---

## Quick Reference

### Start Development Environment

```powershell
# Terminal 1: Backend
cd d:\projects\RLM_demo\rlm-document-retrieval
.\.venv\Scripts\Activate.ps1
uvicorn backend.rlm.api.main:app --reload --port 8000

# Terminal 2: Frontend
cd d:\projects\RLM_demo\rlm-document-retrieval\frontend
npm run dev
```

### Run All Tests

```powershell
.\.venv\Scripts\Activate.ps1
pytest backend/tests/ -v
```

### Check Code Quality

```powershell
ruff check backend/ && ruff format backend/ && mypy backend/rlm
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/docs` | GET | Swagger API documentation |
| `/api/v1/documents/upload` | POST | Upload documents |
| `/api/v1/documents/list` | GET | List uploaded documents |
| `/api/v1/queries/execute` | POST | Execute query with routing |
| `/api/v1/queries/analyze` | POST | Analyze query without executing |
| `/api/v1/trajectory/{session_id}` | GET | Get trajectory data |
| `/ws/trajectory/{session_id}` | WS | Real-time trajectory streaming |

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Resources

- [RLM Paper](https://arxiv.org/pdf/2512.24601) - Original research paper
- [LiteLLM Documentation](https://docs.litellm.ai) - LLM provider integration
- [FastAPI Documentation](https://fastapi.tiangolo.com) - Backend framework
- [React Flow Documentation](https://reactflow.dev) - Tree visualization

---

**Built with the RLM architecture for intelligent document analysis.**
