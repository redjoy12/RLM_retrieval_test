"""Main FastAPI application entry point for RLM Document Retrieval System.

This module initializes the FastAPI application, configures middleware,
mounts static files, and includes all API routes.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from rlm.api.routes import (
    documents_router,
    hybrid_router,
    queries_router,
    sessions_router,
    session_websocket_router,
)
from rlm.api.routes.stats import router as stats_router
from rlm.api.routes.trajectory import router as trajectory_router
from rlm.api.routes.websocket import router as websocket_router
from rlm.config import get_settings
from rlm.config.settings import get_document_settings

# Get settings
settings = get_settings()
doc_settings = get_document_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan context manager.
    
    Handles startup and shutdown events for the application.
    
    Args:
        app: FastAPI application instance
        
    Yields:
        None
    """
    # Startup
    print("[*] Starting RLM Document Retrieval System...")
    
    # Ensure required directories exist
    Path(settings.log_dir).mkdir(parents=True, exist_ok=True)
    Path(doc_settings.storage_path).mkdir(parents=True, exist_ok=True)
    
    print("[+] RLM System initialized successfully")
    
    yield
    
    # Shutdown
    print("[*] Shutting down RLM Document Retrieval System...")
    print("[+] Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="RLM Document Retrieval System",
        description="Recursive Language Model based document retrieval API",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(documents_router, prefix="/api/v1")
    app.include_router(queries_router, prefix="/api/v1")
    app.include_router(sessions_router, prefix="/api/v1")
    app.include_router(hybrid_router, prefix="/api/v1")
    app.include_router(trajectory_router, prefix="/api/v1")
    app.include_router(stats_router, prefix="/api/v1")
    
    # Include WebSocket routes
    app.include_router(websocket_router)
    app.include_router(session_websocket_router)
    
    # Mount static files for frontend (if built)
    frontend_dist = Path(__file__).parent.parent.parent / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")
        
        @app.get("/")
        async def serve_frontend() -> FileResponse:
            """Serve the frontend application."""
            return FileResponse(frontend_dist / "index.html")
    
    return app


# Create the application instance
app = create_app()


@app.get("/api/health")
async def health_check() -> dict:
    """Health check endpoint.
    
    Returns:
        Health status information
    """
    return {
        "status": "healthy",
        "version": "0.1.0",
        "service": "rlm-document-retrieval",
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
