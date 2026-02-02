"""Hybrid RAG Integration - Component 8.

This module provides advanced hybrid RAG capabilities including:
- Hybrid search (semantic + keyword with RRF fusion)
- Cross-encoder reranking
- Adaptive chunk selection
- Citation tracking
- Advanced hybrid strategy for RLM integration

Example:
    ```python
    from rlm.hybrid import HybridSearcher, RerankerPipeline
    from rlm.hybrid import AdvancedHybridStrategy
    
    # Hybrid search
    searcher = HybridSearcher()
    results = await searcher.search("query", document_ids=["doc-1"])
    
    # Advanced strategy
    strategy = AdvancedHybridStrategy()
    result = await strategy.execute("query", document_ids=["doc-1"])
    ```
"""

from rlm.hybrid.search_engines import (
    BM25Searcher,
    HybridSearcher,
    SearchResultAggregator,
)
from rlm.hybrid.reranker import (
    CrossEncoderReranker,
    LLMReranker,
    MultiStageReranker,
    Reranker,
    RerankerPipeline,
)
from rlm.hybrid.chunk_selector import (
    AdaptiveChunkSelector,
    ChunkSelector,
    DiversityBasedSelector,
    ProgressiveSelector,
    ScoreThresholdSelector,
    SelectionStrategyFactory,
)
from rlm.hybrid.citation_manager import (
    Citation,
    CitationManager,
    InlineCitationManager,
)
from rlm.hybrid.advanced_strategy import (
    AdvancedHybridStrategy,
    AdvancedHybridStrategyFactory,
)

__all__ = [
    # Search Engines
    "BM25Searcher",
    "HybridSearcher",
    "SearchResultAggregator",
    # Rerankers
    "CrossEncoderReranker",
    "LLMReranker",
    "MultiStageReranker",
    "Reranker",
    "RerankerPipeline",
    # Chunk Selectors
    "AdaptiveChunkSelector",
    "ChunkSelector",
    "DiversityBasedSelector",
    "ProgressiveSelector",
    "ScoreThresholdSelector",
    "SelectionStrategyFactory",
    # Citation Management
    "Citation",
    "CitationManager",
    "InlineCitationManager",
    # Strategies
    "AdvancedHybridStrategy",
    "AdvancedHybridStrategyFactory",
]

__version__ = "1.0.0"
