"""Data models for query routing and optimization."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class QueryComplexity(Enum):
    """Complexity levels for queries."""
    
    SIMPLE = "simple"           # Direct Q&A, single fact
    SINGLE_HOP = "single_hop"   # One-step retrieval
    MULTI_HOP = "multi_hop"     # Multiple steps, comparisons
    QUADRATIC = "quadratic"     # Complex reasoning, aggregation


class ExecutionStrategy(Enum):
    """Available execution strategies."""
    
    DIRECT_LLM = "direct_llm"       # Simple Q&A with small context
    RAG = "rag"                     # Vector retrieval + single answer
    RLM = "rlm"                     # Full recursive analysis
    HYBRID = "hybrid"               # RAG pre-filter + RLM analysis


@dataclass
class QueryAnalysis:
    """Analysis of a query for routing decisions."""
    
    query: str
    complexity: QueryComplexity
    complexity_score: float  # 0-1 confidence
    keywords: List[str]
    context_needs: str  # "minimal", "moderate", "extensive"
    reasoning_depth: int  # Estimated reasoning steps needed
    is_multi_hop: bool
    requires_comparison: bool
    requires_aggregation: bool
    estimated_chunk_access: int  # Estimated chunks to access


@dataclass
class CostEstimate:
    """Cost estimation for query execution."""
    
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_total_tokens: int
    estimated_cost_usd: float
    model_used: str
    cost_buffer_percent: int = 10
    
    @property
    def cost_with_buffer(self) -> float:
        """Cost with safety buffer applied."""
        return self.estimated_cost_usd * (1 + self.cost_buffer_percent / 100)


@dataclass
class RoutingDecision:
    """Complete routing decision for a query."""
    
    query: str
    strategy: ExecutionStrategy
    confidence: float  # 0-1 confidence score
    reasoning: str  # Human-readable explanation
    query_analysis: QueryAnalysis
    cost_estimate: CostEstimate
    document_ids: List[str]
    context_size: int  # Total characters in context
    estimated_chunks: int  # Estimated chunks to process
    
    # Strategy-specific parameters
    rag_top_k: Optional[int] = None  # For RAG/Hybrid strategies
    hybrid_rag_filter_ratio: Optional[float] = None  # For Hybrid strategy
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "strategy": self.strategy.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "query_analysis": {
                "complexity": self.query_analysis.complexity.value,
                "complexity_score": self.query_analysis.complexity_score,
                "keywords": self.query_analysis.keywords,
                "context_needs": self.query_analysis.context_needs,
                "reasoning_depth": self.query_analysis.reasoning_depth,
                "is_multi_hop": self.query_analysis.is_multi_hop,
                "requires_comparison": self.query_analysis.requires_comparison,
                "requires_aggregation": self.query_analysis.requires_aggregation,
            },
            "cost_estimate": {
                "estimated_input_tokens": self.cost_estimate.estimated_input_tokens,
                "estimated_output_tokens": self.cost_estimate.estimated_output_tokens,
                "estimated_total_tokens": self.cost_estimate.estimated_total_tokens,
                "estimated_cost_usd": round(self.cost_estimate.estimated_cost_usd, 6),
                "cost_with_buffer": round(self.cost_estimate.cost_with_buffer, 6),
                "model_used": self.cost_estimate.model_used,
            },
            "document_ids": self.document_ids,
            "context_size": self.context_size,
            "estimated_chunks": self.estimated_chunks,
            "rag_top_k": self.rag_top_k,
            "hybrid_rag_filter_ratio": self.hybrid_rag_filter_ratio,
        }


@dataclass
class RoutingVisibility:
    """User-facing routing information."""
    
    strategy_name: str
    strategy_description: str
    reasoning: str
    estimated_cost_usd: str
    estimated_tokens: int
    confidence_percent: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "strategy": self.strategy_name,
            "description": self.strategy_description,
            "reasoning": self.reasoning,
            "estimated_cost": self.estimated_cost_usd,
            "estimated_tokens": self.estimated_tokens,
            "confidence": f"{self.confidence_percent}%",
        }


@dataclass
class QuerySuggestion:
    """A suggestion for query optimization."""
    
    suggestion_type: str  # "reformulation", "decomposition", "keywords", "chunking"
    original_query: str
    suggested_query: Optional[str] = None
    explanation: str = ""
    confidence: float = 0.0  # 0-1 confidence
    expected_improvement: Optional[str] = None  # Human-readable improvement description


@dataclass
class OptimizationResult:
    """Results from query optimization."""
    
    original_query: str
    suggestions: List[QuerySuggestion]
    optimized_query: Optional[str] = None
    use_optimized: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_query": self.original_query,
            "optimized_query": self.optimized_query,
            "use_optimized": self.use_optimized,
            "suggestions": [
                {
                    "type": s.suggestion_type,
                    "suggestion": s.suggested_query,
                    "explanation": s.explanation,
                    "confidence": f"{int(s.confidence * 100)}%",
                    "expected_improvement": s.expected_improvement,
                }
                for s in self.suggestions
            ],
        }


@dataclass
class RAGChunk:
    """A chunk retrieved via RAG."""
    
    chunk_id: str
    document_id: str
    content: str
    score: float  # Similarity score
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResult:
    """Result from RAG retrieval."""
    
    query: str
    chunks: List[RAGChunk]
    total_chunks_searched: int
    retrieval_time_ms: float
    
    @property
    def top_chunk(self) -> Optional[RAGChunk]:
        """Get the highest scoring chunk."""
        if not self.chunks:
            return None
        return max(self.chunks, key=lambda c: c.score)
    
    def get_context_string(self, max_chunks: Optional[int] = None) -> str:
        """Get chunks formatted as context string."""
        chunks = self.chunks[:max_chunks] if max_chunks else self.chunks
        return "\n\n---\n\n".join(
            f"[Document: {c.document_id}, Score: {c.score:.3f}]\n{c.content}"
            for c in chunks
        )


@dataclass
class StrategyResult:
    """Result from executing a strategy."""
    
    strategy: ExecutionStrategy
    answer: str
    execution_time_ms: float
    tokens_used: int
    cost_usd: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # RAG-specific
    rag_result: Optional[RAGResult] = None
    
    # RLM-specific
    sub_llm_calls: int = 0
    recursion_depth: int = 0
    trajectory: Optional[List[Dict[str, Any]]] = None


class RoutingError(Exception):
    """Error during query routing."""
    pass


class StrategyError(Exception):
    """Error during strategy execution."""
    pass
