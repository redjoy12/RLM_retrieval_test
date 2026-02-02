"""Cleanup job for managing expired sessions.

Provides automatic cleanup of expired sessions with configurable
scheduling and monitoring.
"""

import asyncio
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import structlog

from rlm.config import get_session_settings
from rlm.session.manager import SessionManager

logger = structlog.get_logger()


class SessionCleanupJob:
    """Background job for cleaning up expired sessions.

    Automatically removes sessions that have exceeded their TTL,
    with configurable scheduling and callbacks.

    Example:
        ```python
        # Start cleanup job
        cleanup = SessionCleanupJob(session_manager)
        await cleanup.start()

        # Later, stop it
        await cleanup.stop()
        ```
    """

    def __init__(
        self,
        session_manager: Optional[SessionManager] = None,
        interval_hours: Optional[int] = None,
        on_cleanup: Optional[Callable[[int], None]] = None,
    ) -> None:
        """Initialize cleanup job.

        Args:
            session_manager: Session manager instance (created if None)
            interval_hours: Cleanup interval in hours (from settings if None)
            on_cleanup: Optional callback when cleanup runs (receives count)
        """
        settings = get_session_settings()

        self.session_manager = session_manager or SessionManager()
        self.interval_hours = interval_hours or settings.cleanup_interval_hours
        self.on_cleanup = on_cleanup

        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._last_run: Optional[datetime] = None
        self._last_count = 0
        self._total_cleaned = 0

        logger.info(
            "cleanup_job_initialized",
            interval_hours=self.interval_hours,
        )

    async def start(self) -> None:
        """Start the cleanup job."""
        if self._running:
            logger.warning("cleanup_job_already_running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())

        logger.info("cleanup_job_started")

    async def stop(self) -> None:
        """Stop the cleanup job."""
        if not self._running:
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("cleanup_job_stopped")

    async def _run_loop(self) -> None:
        """Main cleanup loop."""
        while self._running:
            try:
                # Run cleanup
                count = await self.run_cleanup()

                # Update stats
                self._last_run = datetime.utcnow()
                self._last_count = count
                self._total_cleaned += count

                # Call callback if provided
                if self.on_cleanup:
                    try:
                        self.on_cleanup(count)
                    except Exception as e:
                        logger.error("cleanup_callback_error", error=str(e))

                # Log results
                if count > 0:
                    logger.info(
                        "cleanup_completed",
                        sessions_deleted=count,
                        total_cleaned=self._total_cleaned,
                    )
                else:
                    logger.debug("cleanup_no_expired_sessions")

            except Exception as e:
                logger.error("cleanup_error", error=str(e))

            # Wait for next interval
            if self._running:
                await asyncio.sleep(self.interval_hours * 3600)

    async def run_cleanup(self) -> int:
        """Run a single cleanup operation.

        Returns:
            Number of sessions cleaned up
        """
        return await self.session_manager.cleanup_expired_sessions()

    async def run_once(self) -> int:
        """Run cleanup once (manual trigger).

        Returns:
            Number of sessions cleaned up
        """
        count = await self.run_cleanup()

        self._last_run = datetime.utcnow()
        self._last_count = count
        self._total_cleaned += count

        logger.info("cleanup_manual_run", sessions_deleted=count)

        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cleanup job statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "is_running": self._running,
            "interval_hours": self.interval_hours,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "last_count": self._last_count,
            "total_cleaned": self._total_cleaned,
        }

    async def get_expired_sessions_preview(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Preview sessions that would be cleaned up.

        Args:
            limit: Maximum sessions to preview

        Returns:
            List of session info dictionaries
        """
        from sqlalchemy import select
        from rlm.session.models import Session as SessionModel

        async with self.session_manager.async_session() as session:
            result = await session.execute(
                select(SessionModel).where(SessionModel.expires_at < datetime.utcnow()).limit(limit)
            )

            expired = result.scalars().all()

            return [
                {
                    "id": s.id,
                    "title": s.title,
                    "created_at": s.created_at.isoformat() if s.created_at else None,
                    "expires_at": s.expires_at.isoformat() if s.expires_at else None,
                }
                for s in expired
            ]


class CleanupScheduler:
    """Advanced scheduler for multiple cleanup tasks.

    Manages multiple cleanup jobs with different schedules
    and provides centralized monitoring.

    Example:
        ```python
        scheduler = CleanupScheduler()

        # Add jobs
        scheduler.add_job("sessions", SessionCleanupJob(manager))
        scheduler.add_job("temp_files", TempFileCleanupJob())

        # Start all
        await scheduler.start_all()
        ```
    """

    def __init__(self) -> None:
        """Initialize scheduler."""
        self.jobs: Dict[str, SessionCleanupJob] = {}
        self._running = False

        logger.info("cleanup_scheduler_initialized")

    def add_job(self, name: str, job: SessionCleanupJob) -> None:
        """Add a cleanup job.

        Args:
            name: Job identifier
            job: Cleanup job instance
        """
        self.jobs[name] = job
        logger.info("cleanup_job_added", name=name)

    def remove_job(self, name: str) -> None:
        """Remove a cleanup job.

        Args:
            name: Job identifier
        """
        if name in self.jobs:
            del self.jobs[name]
            logger.info("cleanup_job_removed", name=name)

    async def start_all(self) -> None:
        """Start all cleanup jobs."""
        for name, job in self.jobs.items():
            await job.start()
            logger.info("cleanup_job_started", name=name)

        self._running = True
        logger.info("all_cleanup_jobs_started")

    async def stop_all(self) -> None:
        """Stop all cleanup jobs."""
        for name, job in self.jobs.items():
            await job.stop()
            logger.info("cleanup_job_stopped", name=name)

        self._running = False
        logger.info("all_cleanup_jobs_stopped")

    async def run_all_now(self) -> Dict[str, int]:
        """Manually trigger all cleanup jobs.

        Returns:
            Dictionary of job name -> count cleaned
        """
        results = {}

        for name, job in self.jobs.items():
            try:
                count = await job.run_once()
                results[name] = count
            except Exception as e:
                logger.error("cleanup_job_manual_error", name=name, error=str(e))
                results[name] = -1

        return results

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all jobs.

        Returns:
            Dictionary of job name -> stats
        """
        return {name: job.get_stats() for name, job in self.jobs.items()}


# Convenience functions for common use cases


async def run_session_cleanup_now(
    session_manager: Optional[SessionManager] = None,
) -> int:
    """Run session cleanup immediately.

    Args:
        session_manager: Optional session manager

    Returns:
        Number of sessions cleaned up
    """
    job = SessionCleanupJob(session_manager)
    return await job.run_once()


async def start_background_cleanup(
    session_manager: Optional[SessionManager] = None,
    on_cleanup: Optional[Callable[[int], None]] = None,
) -> SessionCleanupJob:
    """Start background cleanup job.

    Args:
        session_manager: Optional session manager
        on_cleanup: Optional callback

    Returns:
        Running cleanup job
    """
    job = SessionCleanupJob(
        session_manager=session_manager,
        on_cleanup=on_cleanup,
    )
    await job.start()
    return job
