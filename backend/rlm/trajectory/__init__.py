"""Trajectory module for RLM visualization.

Provides trajectory logging, processing, export, and WebSocket streaming
for real-time visualization of RLM execution.
"""

from rlm.trajectory.logger import TrajectoryLogger
from rlm.trajectory.processor import (
    TrajectoryProcessor,
    TrajectoryNode,
    TrajectoryTree,
    TrajectoryStepType,
    TimelineEvent,
    CostBreakdown,
    ExecutionStats,
    TokenCost,
)
from rlm.trajectory.exporter import TrajectoryExporter
from rlm.trajectory.websocket_manager import (
    TrajectoryWebSocketManager,
    TrajectoryStreamAdapter,
    handle_trajectory_websocket,
    websocket_manager,
)

__all__ = [
    "TrajectoryLogger",
    "TrajectoryProcessor",
    "TrajectoryNode",
    "TrajectoryTree",
    "TrajectoryStepType",
    "TimelineEvent",
    "CostBreakdown",
    "ExecutionStats",
    "TokenCost",
    "TrajectoryExporter",
    "TrajectoryWebSocketManager",
    "TrajectoryStreamAdapter",
    "handle_trajectory_websocket",
    "websocket_manager",
]
