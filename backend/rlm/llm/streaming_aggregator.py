"""Streaming aggregator for concurrent sub-LLM call streaming."""

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

import structlog

from rlm.llm.sub_llm_manager import StreamingSubLLMResult, SubLLMCall

logger = structlog.get_logger()


@dataclass
class StreamBuffer:
    """Buffer for collecting streamed chunks from a single call."""

    call_id: str
    query: str
    chunks: List[str] = field(default_factory=list)
    is_complete: bool = False
    error: Optional[str] = None
    started_at: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    last_chunk_at: Optional[float] = None

    def add_chunk(self, chunk: str) -> None:
        """Add a chunk to the buffer."""
        self.chunks.append(chunk)
        self.last_chunk_at = asyncio.get_event_loop().time()

    def get_full_response(self) -> str:
        """Get the complete response from all chunks."""
        return "".join(self.chunks)

    def get_duration_ms(self) -> float:
        """Get streaming duration in milliseconds."""
        if self.last_chunk_at:
            return (self.last_chunk_at - self.started_at) * 1000
        return 0.0


class StreamingAggregator:
    """Aggregates streaming responses from multiple concurrent sub-LLM calls.

    This class manages concurrent streaming calls and yields results as they
    arrive, enabling real-time progress tracking and early result availability.

    Features:
    - Concurrent streaming from multiple calls
    - Real-time result delivery as chunks arrive
    - Progress tracking for each call
    - Error isolation (one failing call doesn't stop others)
    - Buffer management for partial results

    Example:
        ```python
        aggregator = StreamingAggregator()

        calls = [
            SubLLMCall(query="What is X?", context="..."),
            SubLLMCall(query="What is Y?", context="..."),
        ]

        async for result in aggregator.execute(calls, execute_fn, session_id):
            if result.chunk:
                print(f"Chunk from {result.call_id}: {result.chunk}")
            elif result.is_complete:
                print(f"Complete: {result.call_id}")
            elif result.error:
                print(f"Error: {result.error}")
        ```
    """

    def __init__(
        self,
        buffer_size: int = 1000,
        chunk_timeout: float = 30.0,
        enable_interleaving: bool = True,
    ) -> None:
        """Initialize the streaming aggregator.

        Args:
            buffer_size: Maximum chunks to buffer per call
            chunk_timeout: Timeout between chunks in seconds
            enable_interleaving: Whether to interleave chunks from different calls
        """
        self.buffer_size = buffer_size
        self.chunk_timeout = chunk_timeout
        self.enable_interleaving = enable_interleaving

        # Statistics
        self._total_calls = 0
        self._completed_calls = 0
        self._failed_calls = 0
        self._total_chunks = 0

        logger.info(
            "streaming_aggregator_initialized",
            buffer_size=buffer_size,
            chunk_timeout=chunk_timeout,
            interleaving=enable_interleaving,
        )

    async def execute(
        self,
        calls: List[SubLLMCall],
        execute_fn: Callable[[SubLLMCall, str], AsyncIterator[str]],
        session_id: str,
    ) -> AsyncIterator[StreamingSubLLMResult]:
        """Execute multiple calls and stream results as they arrive.

        Args:
            calls: List of SubLLMCall objects to execute
            execute_fn: Function that takes (call, session_id) and yields chunks
            session_id: Session identifier

        Yields:
            StreamingSubLLMResult objects as chunks arrive or calls complete
        """
        if not calls:
            return

        self._total_calls += len(calls)

        logger.info(
            "streaming_execution_started",
            session_id=session_id,
            call_count=len(calls),
        )

        # Create buffers for each call
        buffers: Dict[str, StreamBuffer] = {
            call.id: StreamBuffer(call_id=call.id, query=call.query) for call in calls
        }

        # Create queues for interleaved chunk delivery
        result_queue: asyncio.Queue[StreamingSubLLMResult] = asyncio.Queue()

        # Track active streams
        active_streams: Dict[str, asyncio.Task] = {}

        # Create stream tasks
        for call in calls:
            task = asyncio.create_task(
                self._stream_single_call(
                    call,
                    execute_fn,
                    session_id,
                    buffers[call.id],
                    result_queue,
                )
            )
            active_streams[call.id] = task

        # Process results as they arrive
        completed_streams = 0

        try:
            while completed_streams < len(calls):
                try:
                    # Wait for result with timeout
                    result = await asyncio.wait_for(
                        result_queue.get(),
                        timeout=self.chunk_timeout,
                    )

                    # Update statistics
                    if result.chunk:
                        self._total_chunks += 1

                    if result.is_complete or result.error:
                        completed_streams += 1
                        if result.error:
                            self._failed_calls += 1
                        else:
                            self._completed_calls += 1

                        # Clean up completed stream
                        if result.call_id in active_streams:
                            active_streams[result.call_id].cancel()
                            del active_streams[result.call_id]

                    yield result

                except asyncio.TimeoutError:
                    logger.warning(
                        "streaming_chunk_timeout",
                        session_id=session_id,
                        active_streams=len(active_streams),
                    )
                    break

        finally:
            # Cancel any remaining tasks
            for task in active_streams.values():
                task.cancel()

            logger.info(
                "streaming_execution_complete",
                session_id=session_id,
                completed=self._completed_calls,
                failed=self._failed_calls,
                total_chunks=self._total_chunks,
            )

    async def _stream_single_call(
        self,
        call: SubLLMCall,
        execute_fn: Callable[[SubLLMCall, str], AsyncIterator[str]],
        session_id: str,
        buffer: StreamBuffer,
        result_queue: asyncio.Queue[StreamingSubLLMResult],
    ) -> None:
        """Stream a single call and feed results to the queue.

        Args:
            call: SubLLMCall to execute
            execute_fn: Function that yields chunks
            session_id: Session identifier
            buffer: StreamBuffer for this call
            result_queue: Queue to feed results into
        """
        try:
            async for chunk in execute_fn(call, session_id):
                buffer.add_chunk(chunk)

                await result_queue.put(
                    StreamingSubLLMResult(
                        call_id=call.id,
                        query=call.query,
                        chunk=chunk,
                        metadata={"buffered_chunks": len(buffer.chunks)},
                    )
                )

            # Mark as complete
            buffer.is_complete = True
            await result_queue.put(
                StreamingSubLLMResult(
                    call_id=call.id,
                    query=call.query,
                    is_complete=True,
                    metadata={
                        "total_chunks": len(buffer.chunks),
                        "duration_ms": buffer.get_duration_ms(),
                        "full_response": buffer.get_full_response(),
                    },
                )
            )

        except Exception as e:
            buffer.error = str(e)
            await result_queue.put(
                StreamingSubLLMResult(
                    call_id=call.id,
                    query=call.query,
                    error=str(e),
                    is_complete=True,
                    metadata={"chunks_before_error": len(buffer.chunks)},
                )
            )

    async def execute_sequential(
        self,
        calls: List[SubLLMCall],
        execute_fn: Callable[[SubLLMCall, str], AsyncIterator[str]],
        session_id: str,
    ) -> AsyncIterator[StreamingSubLLMResult]:
        """Execute calls sequentially and stream results.

        Unlike execute(), this processes calls one at a time, which is useful
        when order matters or resource constraints require sequential processing.

        Args:
            calls: List of SubLLMCall objects
            execute_fn: Function that yields chunks
            session_id: Session identifier

        Yields:
            StreamingSubLLMResult objects
        """
        for call in calls:
            buffer = StreamBuffer(call_id=call.id, query=call.query)

            try:
                async for chunk in execute_fn(call, session_id):
                    buffer.add_chunk(chunk)
                    yield StreamingSubLLMResult(
                        call_id=call.id,
                        query=call.query,
                        chunk=chunk,
                    )

                yield StreamingSubLLMResult(
                    call_id=call.id,
                    query=call.query,
                    is_complete=True,
                    metadata={"total_chunks": len(buffer.chunks)},
                )

            except Exception as e:
                yield StreamingSubLLMResult(
                    call_id=call.id,
                    query=call.query,
                    error=str(e),
                    is_complete=True,
                )

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregator statistics.

        Returns:
            Dictionary with streaming statistics
        """
        return {
            "total_calls": self._total_calls,
            "completed_calls": self._completed_calls,
            "failed_calls": self._failed_calls,
            "total_chunks": self._total_chunks,
            "success_rate": (
                self._completed_calls / self._total_calls
                if self._total_calls > 0
                else 0.0
            ),
            "avg_chunks_per_call": (
                self._total_chunks / self._total_calls
                if self._total_calls > 0
                else 0.0
            ),
        }

    def reset_stats(self) -> None:
        """Reset all statistics to zero."""
        self._total_calls = 0
        self._completed_calls = 0
        self._failed_calls = 0
        self._total_chunks = 0
