"""API routes for RLM Document Retrieval System."""

from rlm.api.routes.documents import router as documents_router
from rlm.api.routes.queries import router as queries_router
from rlm.api.routes.hybrid import router as hybrid_router
from rlm.api.routes.sessions import router as sessions_router
from rlm.api.routes.session_websocket import router as session_websocket_router
from rlm.api.routes.stats import router as stats_router
from rlm.api.routes.trajectory import router as trajectory_router
from rlm.api.routes.websocket import router as websocket_router

__all__ = [
    "documents_router",
    "queries_router",
    "hybrid_router",
    "sessions_router",
    "session_websocket_router",
    "stats_router",
    "trajectory_router",
    "websocket_router",
]
