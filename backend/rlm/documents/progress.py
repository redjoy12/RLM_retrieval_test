"""Progress tracking for document processing."""

from datetime import datetime
from typing import Callable, Dict, List, Optional
from uuid import UUID

from rlm.documents.models import ProcessingProgress, ProcessingStatus


class ProgressTracker:
    """Track progress of document processing."""

    def __init__(self) -> None:
        """Initialize progress tracker."""
        self._progress: Dict[UUID, ProcessingProgress] = {}
        self._callbacks: Dict[UUID, List[Callable[[ProcessingProgress], None]]] = {}

    def start_tracking(self, document_id: UUID) -> ProcessingProgress:
        """
        Start tracking a document.

        Args:
            document_id: Document ID

        Returns:
            Initial progress
        """
        progress = ProcessingProgress(
            document_id=document_id,
            status=ProcessingStatus.PENDING,
            stage="initialized",
            progress_percent=0.0,
            message="Document upload started",
        )
        self._progress[document_id] = progress
        self._notify(document_id, progress)
        return progress

    def update_progress(
        self,
        document_id: UUID,
        stage: str,
        progress_percent: float,
        message: str,
    ) -> ProcessingProgress:
        """
        Update processing progress.

        Args:
            document_id: Document ID
            stage: Current processing stage
            progress_percent: Progress percentage (0-100)
            message: Status message

        Returns:
            Updated progress
        """
        if document_id not in self._progress:
            return self.start_tracking(document_id)

        progress = self._progress[document_id]
        progress.stage = stage
        progress.progress_percent = min(100.0, max(0.0, progress_percent))
        progress.message = message

        if stage not in progress.stages_completed:
            progress.stages_completed.append(stage)

        self._notify(document_id, progress)
        return progress

    def mark_completed(self, document_id: UUID) -> ProcessingProgress:
        """
        Mark document processing as completed.

        Args:
            document_id: Document ID

        Returns:
            Final progress
        """
        progress = self._progress.get(document_id)
        if not progress:
            progress = self.start_tracking(document_id)

        progress.status = ProcessingStatus.COMPLETED
        progress.stage = "completed"
        progress.progress_percent = 100.0
        progress.message = "Document processing completed"
        progress.completed_at = datetime.utcnow()

        self._notify(document_id, progress)
        return progress

    def mark_failed(
        self, document_id: UUID, error: str
    ) -> ProcessingProgress:
        """
        Mark document processing as failed.

        Args:
            document_id: Document ID
            error: Error message

        Returns:
            Final progress
        """
        progress = self._progress.get(document_id)
        if not progress:
            progress = self.start_tracking(document_id)

        progress.status = ProcessingStatus.FAILED
        progress.stage = "failed"
        progress.message = f"Processing failed: {error}"
        progress.error = error
        progress.completed_at = datetime.utcnow()

        self._notify(document_id, progress)
        return progress

    def get_progress(self, document_id: UUID) -> Optional[ProcessingProgress]:
        """
        Get current progress for a document.

        Args:
            document_id: Document ID

        Returns:
            Progress or None if not tracking
        """
        return self._progress.get(document_id)

    def subscribe(
        self, document_id: UUID, callback: Callable[[ProcessingProgress], None]
    ) -> None:
        """
        Subscribe to progress updates for a document.

        Args:
            document_id: Document ID
            callback: Callback function for updates
        """
        if document_id not in self._callbacks:
            self._callbacks[document_id] = []
        self._callbacks[document_id].append(callback)

    def unsubscribe(
        self, document_id: UUID, callback: Callable[[ProcessingProgress], None]
    ) -> None:
        """
        Unsubscribe from progress updates.

        Args:
            document_id: Document ID
            callback: Callback function to remove
        """
        if document_id in self._callbacks:
            if callback in self._callbacks[document_id]:
                self._callbacks[document_id].remove(callback)

    def _notify(self, document_id: UUID, progress: ProcessingProgress) -> None:
        """Notify all subscribers of progress update."""
        if document_id in self._callbacks:
            for callback in self._callbacks[document_id]:
                try:
                    callback(progress)
                except Exception:
                    pass

    def clear(self, document_id: UUID) -> None:
        """
        Clear progress tracking for a document.

        Args:
            document_id: Document ID
        """
        if document_id in self._progress:
            del self._progress[document_id]
        if document_id in self._callbacks:
            del self._callbacks[document_id]

    def list_active(self) -> List[ProcessingProgress]:
        """
        List all active (in-progress) tracking.

        Returns:
            List of active progress trackers
        """
        return [
            progress
            for progress in self._progress.values()
            if progress.status in (ProcessingStatus.PENDING, ProcessingStatus.PROCESSING)
        ]
