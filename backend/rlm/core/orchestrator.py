"""RLM Core Orchestrator - Main entry point for RLM execution."""

import asyncio
import time
import uuid
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

import structlog

from rlm.config import get_settings
from rlm.core.exceptions import CodeExecutionError, RecursionLimitError, TimeoutError
from rlm.core.recursion import RecursionController
from rlm.llm.client import LiteLLMClient
from rlm.llm.interface import LLMClientInterface
from rlm.llm.prompts import get_root_system_prompt, get_sub_llm_system_prompt
from rlm.llm.sub_llm_manager import SubLLMManager
from rlm.sandbox.base import REPLSandboxInterface
from rlm.sandbox.local_repl import LocalREPLSandbox
from rlm.trajectory.logger import TrajectoryLogger
from rlm.types import (
    ChunkedContext,
    RLMResult,
    StreamEvent,
    StreamEventType,
    TrajectoryStepType,
)

logger = structlog.get_logger()


class RLMOrchestrator:
    """Main RLM orchestrator that coordinates all components.
    
    This is the core engine that:
    1. Initializes the REPL environment with document context
    2. Calls the Root LLM to generate exploration code
    3. Executes code in the sandbox
    4. Handles sub-LLM calls recursively
    5. Aggregates results and returns the final answer
    
    Example:
        ```python
        orchestrator = RLMOrchestrator()
        result = await orchestrator.execute(
            query="What are the key findings?",
            context="...large document..."
        )
        print(result.answer)
        ```
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClientInterface] = None,
        sandbox: Optional[REPLSandboxInterface] = None,
        recursion_controller: Optional[RecursionController] = None,
        trajectory_logger: Optional[TrajectoryLogger] = None,
        sub_llm_manager: Optional[SubLLMManager] = None,
    ) -> None:
        """Initialize the RLM orchestrator.
        
        Args:
            llm_client: LLM client (creates default if not provided)
            sandbox: REPL sandbox (creates LocalREPL if not provided)
            recursion_controller: Recursion controller (creates default if not provided)
            trajectory_logger: Trajectory logger (creates default if not provided)
            sub_llm_manager: Sub-LLM Manager for async batch processing (creates default if not provided)
        """
        self.settings = get_settings()
        
        # Initialize components
        self.llm_client = llm_client or LiteLLMClient()
        self.sandbox = sandbox or LocalREPLSandbox()
        self.recursion = recursion_controller or RecursionController()
        self.logger = trajectory_logger or TrajectoryLogger()
        
        # Initialize Sub-LLM Manager with the same LLM client
        self.sub_llm_manager = sub_llm_manager or SubLLMManager(
            llm_client=self.llm_client,
            max_concurrent=self.settings.max_sub_llm_calls // 5,  # Conservative default
            enable_caching=True,
        )
        
        logger.info(
            "orchestrator_initialized",
            llm_model=self.llm_client.get_model_name(),
            sandbox_type=self.sandbox.get_sandbox_type(),
            sub_llm_manager_enabled=True,
        )
    
    async def execute(
        self,
        query: str,
        context: str,
        session_id: Optional[str] = None,
        stream_callback: Optional[Callable[[StreamEvent], None]] = None,
    ) -> RLMResult:
        """Execute RLM workflow on a query with document context.
        
        Args:
            query: User query to answer
            context: Document content (can be 10M+ tokens, will be chunked)
            session_id: Optional session ID for tracking
            stream_callback: Optional callback for streaming updates
            
        Returns:
            RLMResult with answer and execution metadata
        """
        start_time = time.time()
        session_id = session_id or str(uuid.uuid4())
        
        # Initialize session
        self.logger.start_session(session_id)
        self.recursion.initialize_root(query)
        
        # Create chunked context for memory efficiency
        chunked_context = ChunkedContext(
            context,
            chunk_size=self.settings.context_chunk_size,
        )
        
        logger.info(
            "rlm_execution_started",
            session_id=session_id,
            query=query[:100],
            context_chunks=chunked_context.total_chunks,
        )
        
        try:
            # Stream start event
            if stream_callback:
                stream_callback(
                    self.logger.create_stream_event(
                        StreamEventType.STEP_START,
                        session_id,
                        {"query": query, "total_chunks": chunked_context.total_chunks},
                    )
                )
            
            # Step 1: Call Root LLM to generate exploration code
            root_response = await self._call_root_llm(
                session_id, query, chunked_context, stream_callback
            )
            
            # Step 2: Execute the generated code
            execution_result = await self._execute_code(
                session_id, root_response, chunked_context, stream_callback
            )
            
            # Step 3: Extract final answer
            final_answer = self._extract_answer(execution_result)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Stream final event
            if stream_callback:
                stream_callback(
                    self.logger.create_stream_event(
                        StreamEventType.FINAL_RESULT,
                        session_id,
                        {"answer": final_answer, "execution_time_ms": execution_time},
                    )
                )
            
            # Get trajectory and stats
            trajectory = self.logger.get_trajectory(session_id)
            stats = self.recursion.get_stats()
            
            result = RLMResult(
                answer=final_answer,
                session_id=session_id,
                trajectory=trajectory,
                total_sub_llm_calls=stats["total_calls"],
                total_tokens_used=0,  # Would need to track from LLM responses
                execution_time_ms=execution_time,
                metadata={
                    "recursion_stats": stats,
                    "context_chunks": chunked_context.total_chunks,
                    "sandbox_type": self.sandbox.get_sandbox_type(),
                },
            )
            
            # End logging session
            self.logger.end_session(session_id, {"result": final_answer})
            
            logger.info(
                "rlm_execution_complete",
                session_id=session_id,
                execution_time_ms=execution_time,
                sub_llm_calls=stats["total_calls"],
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "rlm_execution_failed",
                session_id=session_id,
                error=str(e),
            )
            
            # Log error
            self.logger.log_step(
                session_id,
                TrajectoryStepType.ERROR,
                {"error": str(e), "error_type": type(e).__name__},
            )
            
            self.logger.end_session(session_id, {"error": str(e)})
            
            raise
    
    async def execute_stream(
        self,
        query: str,
        context: str,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Execute RLM with streaming updates.
        
        Args:
            query: User query
            context: Document content
            session_id: Optional session ID
            
        Yields:
            StreamEvent objects with execution updates
        """
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
        
        async def stream_callback(event: StreamEvent) -> None:
            await queue.put(event)
        
        # Run execution in background
        execution_task = asyncio.create_task(
            self.execute(query, context, session_id, stream_callback)
        )
        
        # Yield events as they come
        while not execution_task.done():
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.1)
                yield event
            except asyncio.TimeoutError:
                continue
        
        # Get final result
        try:
            result = await execution_task
            yield self.logger.create_stream_event(
                StreamEventType.FINAL_RESULT,
                result.session_id,
                {
                    "answer": result.answer,
                    "execution_time_ms": result.execution_time_ms,
                    "total_sub_llm_calls": result.total_sub_llm_calls,
                },
            )
        except Exception as e:
            yield self.logger.create_stream_event(
                StreamEventType.ERROR,
                session_id or str(uuid.uuid4()),
                {"error": str(e)},
            )
    
    async def _call_root_llm(
        self,
        session_id: str,
        query: str,
        context: ChunkedContext,
        stream_callback: Optional[Callable[[StreamEvent], None]] = None,
    ) -> str:
        """Call the root LLM to generate exploration code."""
        
        self.logger.log_step(
            session_id,
            TrajectoryStepType.ROOT_LLM_START,
            {"query": query},
        )
        
        # Build system prompt with context summary
        system_prompt = get_root_system_prompt(context.get_summary())
        
        # Build user prompt
        user_prompt = f"""Question: {query}

Write Python code to answer this question by analyzing the document context. 
Remember you have access to:
- `context` variable with the document
- `llm_query(query, chunk)` function for sub-LLM calls

Generate the code now:"""
        
        # Call LLM
        response = await self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=self.settings.default_temperature,
        )
        
        # Log completion
        self.logger.log_step(
            session_id,
            TrajectoryStepType.ROOT_LLM_COMPLETE,
            {
                "generated_code": response.content,
                "model": response.model,
                "usage": response.usage,
            },
        )
        
        # Stream event
        if stream_callback:
            stream_callback(
                self.logger.create_stream_event(
                    StreamEventType.CODE_GENERATED,
                    session_id,
                    {"code": response.content},
                )
            )
        
        return response.content
    
    async def _execute_code(
        self,
        session_id: str,
        code: str,
        context: ChunkedContext,
        stream_callback: Optional[Callable[[StreamEvent], None]] = None,
    ) -> str:
        """Execute generated code in sandbox."""
        
        self.logger.log_step(
            session_id,
            TrajectoryStepType.CODE_EXECUTION_START,
            {"code": code},
        )
        
        # Define sub-LLM callback
        async def sub_llm_callback(query: str, chunk: Optional[str]) -> str:
            return await self._handle_sub_llm_call(session_id, query, chunk, stream_callback)
        
        # Execute in sandbox
        result = await self.sandbox.execute(
            code=code,
            context=context,
            sub_llm_callback=sub_llm_callback,
        )
        
        # Log completion
        self.logger.log_step(
            session_id,
            TrajectoryStepType.CODE_EXECUTION_COMPLETE,
            {
                "output": result.output,
                "error": result.error,
                "execution_time_ms": result.execution_time_ms,
                "sub_llm_calls_count": len(result.sub_llm_calls),
            },
        )
        
        # Stream event
        if stream_callback:
            stream_callback(
                self.logger.create_stream_event(
                    StreamEventType.CODE_OUTPUT,
                    session_id,
                    {"output": result.output, "error": result.error},
                )
            )
        
        if result.error:
            raise CodeExecutionError(result.error, code, result.output)
        
        return result.output
    
    async def _handle_sub_llm_call(
        self,
        parent_session_id: str,
        query: str,
        chunk: Optional[str],
        stream_callback: Optional[Callable[[StreamEvent], None]] = None,
    ) -> str:
        """Handle a sub-LLM call."""
        
        # Check recursion limits
        if not self.recursion.can_spawn_sub_llm(parent_session_id):
            self.logger.log_step(
                parent_session_id,
                TrajectoryStepType.RECURSION_LIMIT_HIT,
                {"query": query},
            )
            return "[Recursion limit reached - cannot process this query]"
        
        # Enter sub-LLM context
        sub_session_id = self.recursion.enter_sub_llm(parent_session_id, query)
        if not sub_session_id:
            return "[Failed to create sub-LLM session]"
        
        self.logger.log_step(
            sub_session_id,
            TrajectoryStepType.SUB_LLM_SPAWN,
            {"query": query, "parent_id": parent_session_id, "chunk": chunk},
        )
        
        # Stream event
        if stream_callback:
            stream_callback(
                self.logger.create_stream_event(
                    StreamEventType.SUB_LLM_SPAWN,
                    sub_session_id,
                    {"query": query, "parent_id": parent_session_id},
                )
            )
        
        try:
            # Use SubLLMManager for the call (with caching, retries, etc.)
            response = await self.sub_llm_manager.call(
                query=query,
                context=chunk,
                session_id=sub_session_id,
                priority=5,
            )
            
            # Log completion
            self.logger.log_step(
                sub_session_id,
                TrajectoryStepType.SUB_LLM_COMPLETE,
                {"response": response, "cached": False},  # Could check from_cache flag
            )
            
            # Stream event
            if stream_callback:
                stream_callback(
                    self.logger.create_stream_event(
                        StreamEventType.SUB_LLM_RESULT,
                        sub_session_id,
                        {"result": response},
                    )
                )
            
            return response
            
        except Exception as e:
            logger.error("sub_llm_call_failed", sub_session_id=sub_session_id, error=str(e))
            return f"[Error in sub-LLM call: {str(e)}]"
        
        finally:
            self.recursion.exit_sub_llm(sub_session_id)
    
    def _extract_answer(self, execution_output: str) -> str:
        """Extract final answer from execution output."""
        # Try to find final_answer variable output
        lines = execution_output.strip().split('\n')
        
        # Look for common patterns
        for i, line in enumerate(lines):
            if 'final_answer' in line.lower() and '=' in line:
                # Extract the value after the equals sign
                parts = line.split('=', 1)
                if len(parts) > 1:
                    return parts[1].strip().strip('"\'')
        
        # If no final_answer found, return the last non-empty line
        for line in reversed(lines):
            stripped = line.strip()
            if stripped:
                return stripped
        
        return execution_output
