"""Trajectory logger for recording RLM execution."""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from rlm.config import get_settings
from rlm.types import StreamEvent, StreamEventType, TrajectoryStepType

logger = structlog.get_logger()


class TrajectoryLogger:
    """Logs RLM execution trajectory to JSONL files.
    
    Each execution produces a trajectory file with detailed information
    about every step: LLM calls, code execution, sub-LLM spawns, errors, etc.
    """
    
    def __init__(self, log_dir: Optional[str] = None) -> None:
        """Initialize the trajectory logger.
        
        Args:
            log_dir: Directory to save trajectory logs (default from settings)
        """
        settings = get_settings()
        
        self.log_dir = Path(log_dir or settings.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = settings.enable_trajectory_logging
        
        # In-memory storage for active sessions
        self._active_sessions: Dict[str, List[Dict]] = {}
        
        logger.info(
            "trajectory_logger_initialized",
            log_dir=str(self.log_dir),
            enabled=self.enabled,
        )
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """Start a new logging session.
        
        Args:
            session_id: Optional session ID (generated if not provided)
            
        Returns:
            Session ID
        """
        session_id = session_id or str(uuid.uuid4())
        self._active_sessions[session_id] = []
        
        logger.debug("trajectory_session_started", session_id=session_id)
        return session_id
    
    def log_step(
        self,
        session_id: str,
        step_type: TrajectoryStepType,
        data: Dict[str, Any],
    ) -> None:
        """Log a single step in the trajectory.
        
        Args:
            session_id: Session ID
            step_type: Type of step
            data: Step data (varies by type)
        """
        if not self.enabled:
            return
        
        step = {
            "timestamp": datetime.utcnow().isoformat(),
            "step_type": step_type.name,
            "session_id": session_id,
            "data": data,
        }
        
        # Add to in-memory session
        if session_id in self._active_sessions:
            self._active_sessions[session_id].append(step)
        
        # Also write immediately to file (for real-time monitoring)
        self._write_step(session_id, step)
        
        logger.debug(
            "trajectory_step_logged",
            session_id=session_id,
            step_type=step_type.name,
        )
    
    def _write_step(self, session_id: str, step: Dict[str, Any]) -> None:
        """Write a step to the trajectory file."""
        try:
            log_file = self.log_dir / f"{session_id}.jsonl"
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(step, default=str) + '\n')
        except Exception as e:
            logger.error(
                "failed_to_write_trajectory_step",
                session_id=session_id,
                error=str(e),
            )
    
    def end_session(self, session_id: str, final_data: Optional[Dict] = None) -> Path:
        """End a logging session and save final data.
        
        Args:
            session_id: Session ID
            final_data: Optional final result data
            
        Returns:
            Path to the trajectory file
        """
        if not self.enabled:
            return Path()
        
        log_file = self.log_dir / f"{session_id}.jsonl"
        
        # Write final step
        if final_data:
            final_step = {
                "timestamp": datetime.utcnow().isoformat(),
                "step_type": TrajectoryStepType.FINAL_ANSWER.name,
                "session_id": session_id,
                "data": final_data,
            }
            self._write_step(session_id, final_step)
        
        # Clean up in-memory session
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]
        
        logger.info(
            "trajectory_session_ended",
            session_id=session_id,
            log_file=str(log_file),
        )
        
        return log_file
    
    def get_trajectory(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve full trajectory for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            List of trajectory steps
        """
        log_file = self.log_dir / f"{session_id}.jsonl"
        
        if not log_file.exists():
            return []
        
        trajectory = []
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        trajectory.append(json.loads(line))
        except Exception as e:
            logger.error(
                "failed_to_read_trajectory",
                session_id=session_id,
                error=str(e),
            )
        
        return trajectory
    
    def create_stream_event(
        self,
        event_type: StreamEventType,
        session_id: str,
        data: Dict[str, Any],
    ) -> StreamEvent:
        """Create a streaming event.
        
        Args:
            event_type: Type of event
            session_id: Session ID
            data: Event data
            
        Returns:
            StreamEvent object
        """
        return StreamEvent(
            type=event_type,
            session_id=session_id,
            data=data,
        )
