"""Type definitions for RLM Document Retrieval System."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Union,
)


class TrajectoryStepType(Enum):
    """Types of steps in an RLM trajectory."""
    
    ROOT_LLM_START = auto()
    ROOT_LLM_COMPLETE = auto()
    CODE_GENERATED = auto()
    CODE_EXECUTION_START = auto()
    CODE_EXECUTION_COMPLETE = auto()
    SUB_LLM_SPAWN = auto()
    SUB_LLM_COMPLETE = auto()
    ERROR = auto()
    RECURSION_LIMIT_HIT = auto()
    TIMEOUT = auto()
    FINAL_ANSWER = auto()


class StreamEventType(Enum):
    """Types of streaming events."""
    
    STEP_START = "step_start"
    STEP_END = "step_end"
    CODE_GENERATED = "code_generated"
    CODE_OUTPUT = "code_output"
    SUB_LLM_SPAWN = "sub_llm_spawn"
    SUB_LLM_RESULT = "sub_llm_result"
    LLM_TOKEN = "llm_token"
    ERROR = "error"
    PROGRESS = "progress"
    FINAL_RESULT = "final_result"


@dataclass
class StreamEvent:
    """Event for streaming updates."""
    
    type: StreamEventType
    session_id: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class REPLResult:
    """Result from REPL code execution."""
    
    output: str
    error: Optional[str] = None
    sub_llm_calls: List[Dict[str, Any]] = field(default_factory=list)
    execution_time_ms: float = 0.0
    memory_usage_mb: float = 0.0


@dataclass
class SubLLMCall:
    """A sub-LLM call request."""
    
    query: str
    context_chunk: Optional[str] = None
    parent_session_id: str = ""
    depth: int = 0
    call_id: str = field(default_factory=lambda: str(id(object())))


@dataclass
class RLMResult:
    """Final result from RLM execution."""
    
    answer: str
    session_id: str
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    total_sub_llm_calls: int = 0
    total_tokens_used: int = 0
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChunkedContext:
    """Memory-efficient context for huge documents (10M+ tokens)."""
    
    def __init__(self, content: str, chunk_size: int = 100000) -> None:
        """Initialize chunked context.
        
        Args:
            content: Full document content
            chunk_size: Approximate characters per chunk (default 100K chars ~ 25K tokens)
        """
        self._content = content
        self._chunk_size = chunk_size
        self._chunks: List[str] = []
        self._chunk_count = 0
        self._total_length = len(content)
        
        # Pre-compute chunk boundaries
        self._compute_chunks()
    
    def _compute_chunks(self) -> None:
        """Compute chunk boundaries without loading all chunks into memory."""
        if not self._content:
            return
            
        # Split at paragraph boundaries when possible
        paragraphs = self._content.split('\n\n')
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            if current_size + para_size > self._chunk_size and current_chunk:
                # Save current chunk
                self._chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size
        
        # Don't forget the last chunk
        if current_chunk:
            self._chunks.append('\n\n'.join(current_chunk))
        
        self._chunk_count = len(self._chunks)
    
    @property
    def total_chunks(self) -> int:
        """Get total number of chunks."""
        return self._chunk_count
    
    @property
    def total_length(self) -> int:
        """Get total content length."""
        return self._total_length
    
    def get_chunk(self, index: int) -> str:
        """Get a specific chunk by index.
        
        Args:
            index: Chunk index (0-based)
            
        Returns:
            Chunk content
            
        Raises:
            IndexError: If index is out of range
        """
        if index < 0 or index >= self._chunk_count:
            raise IndexError(f"Chunk index {index} out of range (0-{self._chunk_count-1})")
        
        return self._chunks[index]
    
    def get_chunk_range(self, start: int, end: int) -> str:
        """Get multiple chunks as a single string.
        
        Args:
            start: Start chunk index (inclusive)
            end: End chunk index (exclusive)
            
        Returns:
            Concatenated chunk content
        """
        if start < 0:
            start = 0
        if end > self._chunk_count:
            end = self._chunk_count
        
        return '\n\n'.join(self._chunks[start:end])
    
    def search(self, pattern: str) -> List[int]:
        """Search for pattern and return chunk indices where it appears.
        
        Args:
            pattern: Search pattern (case-insensitive)
            
        Returns:
            List of chunk indices containing the pattern
        """
        pattern_lower = pattern.lower()
        matching_chunks = []
        
        for i, chunk in enumerate(self._chunks):
            if pattern_lower in chunk.lower():
                matching_chunks.append(i)
        
        return matching_chunks
    
    def get_summary(self) -> Dict[str, Any]:
        """Get context summary for LLM."""
        return {
            "total_chunks": self._chunk_count,
            "total_characters": self._total_length,
            "chunk_size": self._chunk_size,
            "average_chunk_size": self._total_length // max(self._chunk_count, 1),
        }


# Type aliases
SubLLMCallback = Callable[[SubLLMCall], Any]
StreamCallback = Callable[[StreamEvent], None]
