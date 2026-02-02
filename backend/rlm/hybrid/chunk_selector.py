"""Chunk selection strategies for adaptive retrieval.

This module provides intelligent chunk selection strategies that adapt to
query complexity and result characteristics to optimize retrieval quality.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import structlog
import numpy as np

logger = structlog.get_logger()


class ChunkSelector:
    """Base class for chunk selection strategies."""
    
    def select(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
        max_chunks: int = 10,
    ) -> List[Dict[str, Any]]:
        """Select chunks based on strategy.
        
        Args:
            chunks: List of scored chunk dictionaries
            query: Original search query
            max_chunks: Maximum number of chunks to select
            
        Returns:
            Selected chunks
        """
        raise NotImplementedError


class AdaptiveChunkSelector(ChunkSelector):
    """Adaptive chunk selector that adjusts selection based on query and results.
    
    Considers multiple factors:
    - Query complexity (length, keywords)
    - Score distribution (confidence in results)
    - Content diversity (avoid redundant chunks)
    - Token budget constraints
    
    Example:
        ```python
        selector = AdaptiveChunkSelector(
            min_chunks=3,
            max_chunks=10,
            diversity_threshold=0.8,
        )
        
        selected = selector.select(
            chunks=scored_chunks,
            query="What is machine learning?",
            max_chunks=8,
        )
        ```
    """
    
    def __init__(
        self,
        min_chunks: int = 3,
        max_chunks: int = 10,
        diversity_threshold: float = 0.8,
        score_gap_threshold: float = 0.2,
        token_budget: Optional[int] = None,
        enable_deduplication: bool = True,
    ) -> None:
        """Initialize adaptive chunk selector.
        
        Args:
            min_chunks: Minimum chunks to select
            max_chunks: Maximum chunks to select
            diversity_threshold: Minimum cosine distance for diversity
            score_gap_threshold: Gap between scores to stop selection
            token_budget: Optional token budget constraint
            enable_deduplication: Whether to remove near-duplicate chunks
        """
        self.min_chunks = min_chunks
        self.max_chunks = max_chunks
        self.diversity_threshold = diversity_threshold
        self.score_gap_threshold = score_gap_threshold
        self.token_budget = token_budget
        self.enable_deduplication = enable_deduplication
        
        logger.info(
            "adaptive_chunk_selector_initialized",
            min_chunks=min_chunks,
            max_chunks=max_chunks,
            diversity_threshold=diversity_threshold,
        )
    
    def select(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
        max_chunks: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Adaptively select chunks.
        
        Args:
            chunks: List of scored chunk dictionaries
            query: Original search query
            max_chunks: Override max chunks (optional)
            
        Returns:
            Adaptively selected chunks
        """
        if not chunks:
            return []
        
        max_c = max_chunks or self.max_chunks
        
        # Estimate query complexity
        query_complexity = self._estimate_query_complexity(query)
        
        # Adjust target chunk count based on complexity
        target_count = self._compute_target_count(
            len(chunks),
            query_complexity,
            max_c,
        )
        
        # Sort by score (descending)
        sorted_chunks = sorted(
            chunks,
            key=lambda x: x.get("score", 0) or x.get("cross_encoder_score", 0) or x.get("llm_score", 0),
            reverse=True,
        )
        
        # Remove near-duplicates if enabled
        if self.enable_deduplication:
            sorted_chunks = self._deduplicate_chunks(sorted_chunks)
        
        # Select chunks with diversity
        selected = self._select_with_diversity(
            sorted_chunks,
            target_count,
        )
        
        logger.info(
            "adaptive_selection_complete",
            input_chunks=len(chunks),
            selected_chunks=len(selected),
            query_complexity=query_complexity,
            target_count=target_count,
        )
        
        return selected
    
    def _estimate_query_complexity(self, query: str) -> float:
        """Estimate query complexity on a scale of 0-1.
        
        Args:
            query: Search query
            
        Returns:
            Complexity score (0-1)
        """
        complexity = 0.5  # Base complexity
        
        # Factor 1: Query length
        word_count = len(query.split())
        if word_count > 20:
            complexity += 0.2
        elif word_count > 10:
            complexity += 0.1
        elif word_count < 5:
            complexity -= 0.1
        
        # Factor 2: Complex keywords
        complex_keywords = [
            "compare", "contrast", "difference", "similarities",
            "analyze", "evaluate", "assess", "relationship",
            "cause", "effect", "impact", "influence",
            "trend", "pattern", "correlation", "summary",
        ]
        query_lower = query.lower()
        for keyword in complex_keywords:
            if keyword in query_lower:
                complexity += 0.05
        
        # Factor 3: Question words that imply complexity
        complex_questions = ["how", "why", "explain", "what are the", "describe"]
        for qword in complex_questions:
            if query_lower.startswith(qword):
                complexity += 0.05
        
        return min(1.0, max(0.0, complexity))
    
    def _compute_target_count(
        self,
        available_chunks: int,
        query_complexity: float,
        max_chunks: int,
    ) -> int:
        """Compute target number of chunks based on complexity.
        
        Args:
            available_chunks: Total available chunks
            query_complexity: Query complexity score (0-1)
            max_chunks: Maximum allowed chunks
            
        Returns:
            Target chunk count
        """
        # Base: 30% of available, adjusted by complexity
        base_count = int(available_chunks * (0.3 + query_complexity * 0.4))
        
        # Apply bounds
        target = max(self.min_chunks, min(base_count, max_chunks, available_chunks))
        
        return target
    
    def _deduplicate_chunks(
        self,
        chunks: List[Dict[str, Any]],
        similarity_threshold: float = 0.85,
    ) -> List[Dict[str, Any]]:
        """Remove near-duplicate chunks based on content similarity.
        
        Args:
            chunks: List of chunks
            similarity_threshold: Jaccard similarity threshold for deduplication
            
        Returns:
            Deduplicated chunks
        """
        if not chunks:
            return []
        
        deduplicated = [chunks[0]]  # Keep first chunk
        
        for chunk in chunks[1:]:
            is_duplicate = False
            
            for kept_chunk in deduplicated:
                similarity = self._compute_jaccard_similarity(
                    chunk.get("content", ""),
                    kept_chunk.get("content", ""),
                )
                if similarity > similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(chunk)
        
        logger.debug(
            "deduplication_complete",
            input_count=len(chunks),
            output_count=len(deduplicated),
        )
        
        return deduplicated
    
    def _compute_jaccard_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Jaccard similarity (0-1)
        """
        # Simple word-based Jaccard
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _select_with_diversity(
        self,
        chunks: List[Dict[str, Any]],
        target_count: int,
    ) -> List[Dict[str, Any]]:
        """Select chunks ensuring content diversity.
        
        Args:
            chunks: Sorted chunks by relevance
            target_count: Target number of chunks
            
        Returns:
            Diverse selection of chunks
        """
        if not chunks:
            return []
        
        selected = [chunks[0]]  # Always select top result
        
        for chunk in chunks[1:]:
            if len(selected) >= target_count:
                break
            
            # Check if diverse enough from already selected
            is_diverse = self._check_diversity(chunk, selected)
            
            if is_diverse:
                selected.append(chunk)
        
        # If we haven't met min_chunks, add more even if not diverse
        for chunk in chunks[1:]:
            if len(selected) >= self.min_chunks:
                break
            if chunk not in selected:
                selected.append(chunk)
        
        return selected
    
    def _check_diversity(
        self,
        chunk: Dict[str, Any],
        selected: List[Dict[str, Any]],
    ) -> bool:
        """Check if chunk is diverse enough from selected chunks.
        
        Args:
            chunk: Chunk to check
            selected: Already selected chunks
            
        Returns:
            True if diverse enough
        """
        for sel_chunk in selected:
            similarity = self._compute_jaccard_similarity(
                chunk.get("content", ""),
                sel_chunk.get("content", ""),
            )
            
            # If too similar to any selected chunk, reject
            if similarity > self.diversity_threshold:
                return False
        
        return True
    
    def get_selection_stats(
        self,
        chunks: List[Dict[str, Any]],
        selected: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Get statistics about chunk selection.
        
        Args:
            chunks: Original chunks
            selected: Selected chunks
            
        Returns:
            Selection statistics
        """
        if not chunks or not selected:
            return {
                "compression_ratio": 0.0,
                "avg_score_original": 0.0,
                "avg_score_selected": 0.0,
            }
        
        def get_score(c):
            return c.get("score", 0) or c.get("cross_encoder_score", 0) or c.get("llm_score", 0)
        
        original_scores = [get_score(c) for c in chunks]
        selected_scores = [get_score(c) for c in selected]
        
        return {
            "compression_ratio": len(selected) / len(chunks),
            "avg_score_original": sum(original_scores) / len(original_scores),
            "avg_score_selected": sum(selected_scores) / len(selected_scores),
            "score_retention": sum(selected_scores) / sum(original_scores) if sum(original_scores) > 0 else 0,
        }


class DiversityBasedSelector(ChunkSelector):
    """Select chunks maximizing content diversity using clustering.
    
    Uses K-means clustering to select representative chunks from
    different content clusters, ensuring broad coverage.
    
    Example:
        ```python
        selector = DiversityBasedSelector(n_clusters=5)
        selected = selector.select(chunks, query, max_chunks=8)
        ```
    """
    
    def __init__(
        self,
        n_clusters: int = 5,
        min_chunks_per_cluster: int = 1,
        max_chunks: int = 10,
    ) -> None:
        """Initialize diversity-based selector.
        
        Args:
            n_clusters: Number of content clusters
            min_chunks_per_cluster: Minimum chunks to select per cluster
            max_chunks: Maximum total chunks to select
        """
        self.n_clusters = n_clusters
        self.min_chunks_per_cluster = min_chunks_per_cluster
        self.max_chunks = max_chunks
        
        logger.info(
            "diversity_selector_initialized",
            n_clusters=n_clusters,
            min_per_cluster=min_chunks_per_cluster,
        )
    
    def select(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
        max_chunks: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Select diverse chunks using clustering.
        
        Args:
            chunks: List of scored chunks
            query: Search query
            max_chunks: Override max chunks
            
        Returns:
            Diverse selection of chunks
        """
        if not chunks:
            return []
        
        max_c = max_chunks or self.max_chunks
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Extract content
            contents = [c.get("content", "") for c in chunks]
            
            # Vectorize
            vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
            vectors = vectorizer.fit_transform(contents)
            
            # Cluster
            n_clusters = min(self.n_clusters, len(chunks))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(vectors)
            
            # Select top-scored chunk from each cluster
            selected = []
            for cluster_id in range(n_clusters):
                cluster_chunks = [
                    (i, chunks[i]) for i, label in enumerate(cluster_labels)
                    if label == cluster_id
                ]
                
                if cluster_chunks:
                    # Sort by score and take best from cluster
                    cluster_chunks.sort(
                        key=lambda x: x[1].get("score", 0),
                        reverse=True,
                    )
                    selected.extend([c[1] for c in cluster_chunks[:self.min_chunks_per_cluster]])
            
            # Limit to max_chunks
            selected = selected[:max_c]
            
            logger.info(
                "diversity_selection_complete",
                n_clusters=n_clusters,
                selected=len(selected),
            )
            
            return selected
            
        except ImportError:
            logger.warning("sklearn_not_available, falling back to simple selection")
            # Fallback: simple top-k selection
            sorted_chunks = sorted(
                chunks,
                key=lambda x: x.get("score", 0),
                reverse=True,
            )
            return sorted_chunks[:max_c]


class ScoreThresholdSelector(ChunkSelector):
    """Select chunks based on absolute score thresholds.
    
    Selects chunks that meet minimum score requirements, useful when
    you want to ensure high relevance.
    
    Example:
        ```python
        selector = ScoreThresholdSelector(
            min_score=0.7,
            max_chunks=10,
        )
        selected = selector.select(chunks, query)
        ```
    """
    
    def __init__(
        self,
        min_score: float = 0.6,
        max_chunks: int = 10,
        relative_threshold: bool = False,
    ) -> None:
        """Initialize score threshold selector.
        
        Args:
            min_score: Minimum score threshold
            max_chunks: Maximum chunks to select
            relative_threshold: If True, threshold is relative to top score
        """
        self.min_score = min_score
        self.max_chunks = max_chunks
        self.relative_threshold = relative_threshold
        
        logger.info(
            "threshold_selector_initialized",
            min_score=min_score,
            relative_threshold=relative_threshold,
        )
    
    def select(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
        max_chunks: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Select chunks meeting score threshold.
        
        Args:
            chunks: List of scored chunks
            query: Search query
            max_chunks: Override max chunks
            
        Returns:
            Chunks meeting threshold
        """
        if not chunks:
            return []
        
        max_c = max_chunks or self.max_chunks
        
        # Get scores
        def get_score(c):
            return c.get("score", 0) or c.get("cross_encoder_score", 0) or c.get("llm_score", 0)
        
        scores = [get_score(c) for c in chunks]
        
        if not scores:
            return []
        
        # Compute threshold
        if self.relative_threshold:
            max_score = max(scores)
            threshold = max_score * self.min_score
        else:
            threshold = self.min_score
        
        # Select chunks meeting threshold
        selected = [
            chunk for chunk, score in zip(chunks, scores)
            if score >= threshold
        ][:max_c]
        
        logger.info(
            "threshold_selection_complete",
            threshold=threshold,
            selected=len(selected),
            available=len(chunks),
        )
        
        return selected


class ProgressiveSelector(ChunkSelector):
    """Progressively select chunks until quality criteria are met.
    
    Adds chunks one at a time until either:
    - Score improvement drops below threshold
    - Max chunks reached
    - Content coverage is sufficient
    
    Example:
        ```python
        selector = ProgressiveSelector(
            min_improvement=0.01,
            max_chunks=10,
        )
        selected = selector.select(chunks, query)
        ```
    """
    
    def __init__(
        self,
        min_improvement: float = 0.01,
        max_chunks: int = 10,
        coverage_threshold: float = 0.9,
    ) -> None:
        """Initialize progressive selector.
        
        Args:
            min_improvement: Minimum score improvement to add another chunk
            max_chunks: Maximum chunks to select
            coverage_threshold: Content coverage target (not yet implemented)
        """
        self.min_improvement = min_improvement
        self.max_chunks = max_chunks
        self.coverage_threshold = coverage_threshold
        
        logger.info(
            "progressive_selector_initialized",
            min_improvement=min_improvement,
            max_chunks=max_chunks,
        )
    
    def select(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
        max_chunks: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Progressively select chunks.
        
        Args:
            chunks: List of scored chunks (sorted by score)
            query: Search query
            max_chunks: Override max chunks
            
        Returns:
            Progressively selected chunks
        """
        if not chunks:
            return []
        
        max_c = max_chunks or self.max_chunks
        max_c = min(max_c, len(chunks))
        
        # Ensure chunks are sorted
        sorted_chunks = sorted(
            chunks,
            key=lambda x: x.get("score", 0) or x.get("cross_encoder_score", 0) or x.get("llm_score", 0),
            reverse=True,
        )
        
        def get_score(c):
            return c.get("score", 0) or c.get("cross_encoder_score", 0) or c.get("llm_score", 0)
        
        selected = [sorted_chunks[0]]  # Always include top result
        cumulative_score = get_score(sorted_chunks[0])
        
        for chunk in sorted_chunks[1:]:
            if len(selected) >= max_c:
                break
            
            score = get_score(chunk)
            improvement = score / cumulative_score if cumulative_score > 0 else 0
            
            if improvement >= self.min_improvement:
                selected.append(chunk)
                cumulative_score += score
            else:
                # Stop if improvement is too small
                break
        
        logger.info(
            "progressive_selection_complete",
            selected=len(selected),
            cumulative_score=cumulative_score,
        )
        
        return selected


class SelectionStrategyFactory:
    """Factory for creating chunk selection strategies.
    
    Example:
        ```python
        selector = SelectionStrategyFactory.create("adaptive")
        selector = SelectionStrategyFactory.create("diversity", n_clusters=5)
        ```
    """
    
    @staticmethod
    def create(
        strategy: str,
        **kwargs: Any,
    ) -> ChunkSelector:
        """Create a chunk selector.
        
        Args:
            strategy: Strategy name ("adaptive", "diversity", "threshold", "progressive")
            **kwargs: Strategy-specific parameters
            
        Returns:
            Chunk selector instance
        """
        strategies = {
            "adaptive": AdaptiveChunkSelector,
            "diversity": DiversityBasedSelector,
            "threshold": ScoreThresholdSelector,
            "progressive": ProgressiveSelector,
        }
        
        selector_class = strategies.get(strategy)
        if not selector_class:
            raise ValueError(f"Unknown selection strategy: {strategy}")
        
        return selector_class(**kwargs)
