"""Reranking modules for improving retrieval quality.

This module provides reranking capabilities using cross-encoder models and
LLM-based reranking to improve the precision of retrieved chunks.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import structlog
import numpy as np

logger = structlog.get_logger()


class Reranker(ABC):
    """Abstract base class for rerankers."""
    
    @abstractmethod
    async def rerank(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank chunks based on relevance to query.
        
        Args:
            query: The search query
            chunks: List of chunk dictionaries with 'content' field
            top_k: Number of top results to return (default: all)
            
        Returns:
            Reranked chunks with updated scores
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Reranker name."""
        pass


class CrossEncoderReranker(Reranker):
    """Cross-encoder based reranker for high-precision relevance scoring.
    
    Uses a cross-encoder model (like ms-marco-MiniLM) to score query-chunk pairs,
    providing more accurate relevance scores than bi-encoders.
    
    Example:
        ```python
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        
        scored_chunks = await reranker.rerank(
            query="What is machine learning?",
            chunks=[{"content": "..."}, {"content": "..."}],
            top_k=5
        )
        ```
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        max_length: int = 512,
        device: str = "cpu",
    ) -> None:
        """Initialize cross-encoder reranker.
        
        Args:
            model_name: HuggingFace model name for cross-encoder
            batch_size: Batch size for inference
            max_length: Maximum sequence length
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        
        self._model = None
        self._tokenizer = None
        
        logger.info(
            "cross_encoder_reranker_initialized",
            model=model_name,
            batch_size=batch_size,
            device=device,
        )
    
    def _load_model(self) -> None:
        """Lazy load the cross-encoder model."""
        if self._model is not None:
            return
        
        try:
            from sentence_transformers import CrossEncoder
            
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_length,
            )
            
            logger.info(
                "cross_encoder_model_loaded",
                model=self.model_name,
                device=self.device,
            )
        except ImportError:
            logger.error(
                "sentence_transformers_not_installed",
                message="Install with: pip install sentence-transformers",
            )
            raise
        except Exception as e:
            logger.error("cross_encoder_load_failed", error=str(e))
            raise
    
    async def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank chunks using cross-encoder.
        
        Args:
            query: Search query
            chunks: List of chunk dictionaries (must have 'content' field)
            top_k: Number of top results to return
            
        Returns:
            Reranked chunks with cross-encoder scores
        """
        if not chunks:
            return []
        
        # Load model if needed
        if self._model is None:
            self._load_model()
        
        # Prepare query-chunk pairs
        pairs = [(query, chunk.get("content", "")) for chunk in chunks]
        
        # Score in batches (run in thread pool to avoid blocking)
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            self._score_pairs,
            pairs,
        )
        
        # Add scores to chunks
        for chunk, score in zip(chunks, scores):
            chunk["cross_encoder_score"] = float(score)
            chunk["reranker"] = self.name
        
        # Sort by cross-encoder score
        reranked = sorted(
            chunks,
            key=lambda x: x["cross_encoder_score"],
            reverse=True,
        )
        
        # Update rank
        for i, chunk in enumerate(reranked):
            chunk["rerank"] = i + 1
        
        # Return top_k if specified
        if top_k:
            reranked = reranked[:top_k]
        
        logger.info(
            "cross_encoder_rerank_complete",
            query=query[:50],
            chunks_processed=len(chunks),
            top_score=reranked[0]["cross_encoder_score"] if reranked else 0,
        )
        
        return reranked
    
    def _score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Score query-chunk pairs using cross-encoder.
        
        Args:
            pairs: List of (query, chunk) tuples
            
        Returns:
            List of relevance scores
        """
        all_scores = []
        
        # Process in batches
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            batch_scores = self._model.predict(
                batch,
                show_progress_bar=False,
            )
            all_scores.extend(batch_scores.tolist())
        
        return all_scores
    
    @property
    def name(self) -> str:
        """Reranker name."""
        return f"cross-encoder-{self.model_name.split('/')[-1]}"


class LLMReranker(Reranker):
    """LLM-based reranker for high-accuracy relevance scoring.
    
    Uses an LLM to score chunk relevance to the query. More expensive but
    potentially more accurate than cross-encoders for complex queries.
    
    Example:
        ```python
        reranker = LLMReranker(
            model="gpt-5-mini",
            temperature=0.0,
        )
        
        scored_chunks = await reranker.rerank(
            query="What is machine learning?",
            chunks=chunks,
            top_k=5
        )
        ```
    """
    
    def __init__(
        self,
        model: str = "gpt-5-mini",
        temperature: float = 0.0,
        max_tokens: int = 10,
        max_chunks: int = 20,
        score_format: str = "numeric",  # "numeric" or "categorical"
    ) -> None:
        """Initialize LLM reranker.
        
        Args:
            model: LLM model to use
            temperature: Sampling temperature (0 for deterministic)
            max_tokens: Max tokens for score output
            max_chunks: Maximum chunks to rerank (for cost control)
            score_format: Format of scores - "numeric" (0-10) or "categorical"
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_chunks = max_chunks
        self.score_format = score_format
        
        logger.info(
            "llm_reranker_initialized",
            model=model,
            score_format=score_format,
            max_chunks=max_chunks,
        )
    
    async def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank chunks using LLM scoring.
        
        Args:
            query: Search query
            chunks: List of chunk dictionaries
            top_k: Number of top results to return
            
        Returns:
            Reranked chunks with LLM scores
        """
        if not chunks:
            return []
        
        # Limit chunks for cost control
        chunks_to_score = chunks[:self.max_chunks]
        
        # Score chunks concurrently
        tasks = [
            self._score_chunk(query, chunk)
            for chunk in chunks_to_score
        ]
        scores = await asyncio.gather(*tasks)
        
        # Add scores to chunks
        for chunk, score in zip(chunks_to_score, scores):
            chunk["llm_score"] = score
            chunk["reranker"] = self.name
        
        # Sort by LLM score
        reranked = sorted(
            chunks_to_score,
            key=lambda x: x["llm_score"],
            reverse=True,
        )
        
        # Update rank
        for i, chunk in enumerate(reranked):
            chunk["rerank"] = i + 1
        
        # Return top_k if specified
        if top_k:
            reranked = reranked[:top_k]
        
        logger.info(
            "llm_rerank_complete",
            query=query[:50],
            chunks_scored=len(chunks_to_score),
            top_score=reranked[0]["llm_score"] if reranked else 0,
        )
        
        return reranked
    
    async def _score_chunk(self, query: str, chunk: Dict[str, Any]) -> float:
        """Score a single chunk using LLM.
        
        Args:
            query: Search query
            chunk: Chunk dictionary
            
        Returns:
            Relevance score (0-10)
        """
        from rlm.llm.client import LiteLLMClient
        
        content = chunk.get("content", "")[:1000]  # Limit content length
        
        if self.score_format == "numeric":
            prompt = f"""Rate the relevance of the following text to the query on a scale of 0-10.

Query: {query}

Text: {content}

Provide only a number from 0 to 10, where 10 means perfectly relevant and 0 means completely irrelevant.

Score:"""
        else:
            prompt = f"""Rate the relevance of the following text to the query.

Query: {query}

Text: {content}

Respond with exactly one of: HIGH, MEDIUM, LOW

Relevance:"""
        
        try:
            client = LiteLLMClient(model=self.model)
            response = await client.generate(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            score_text = response.content.strip()
            
            if self.score_format == "numeric":
                # Extract numeric score
                score = self._parse_numeric_score(score_text)
            else:
                # Parse categorical
                score = self._parse_categorical_score(score_text)
            
            return score
            
        except Exception as e:
            logger.error(
                "llm_score_failed",
                chunk_id=chunk.get("chunk_id", "unknown"),
                error=str(e),
            )
            return 0.0
    
    def _parse_numeric_score(self, text: str) -> float:
        """Parse numeric score from LLM output.
        
        Args:
            text: LLM output text
            
        Returns:
            Numeric score 0-10
        """
        # Extract first number from text
        import re
        match = re.search(r'\d+\.?\d*', text)
        if match:
            score = float(match.group())
            return max(0, min(10, score))  # Clamp to 0-10
        return 5.0  # Default middle score
    
    def _parse_categorical_score(self, text: str) -> float:
        """Parse categorical score from LLM output.
        
        Args:
            text: LLM output text
            
        Returns:
            Numeric score 0-10
        """
        text_upper = text.upper()
        if "HIGH" in text_upper:
            return 8.0
        elif "MEDIUM" in text_upper:
            return 5.0
        elif "LOW" in text_upper:
            return 2.0
        return 5.0  # Default
    
    @property
    def name(self) -> str:
        """Reranker name."""
        return f"llm-{self.model}"


class MultiStageReranker(Reranker):
    """Multi-stage reranking pipeline combining multiple rerankers.
    
    Applies multiple rerankers in sequence, progressively refining results.
    Useful for combining fast and accurate reranking methods.
    
    Example:
        ```python
        # Stage 1: Fast cross-encoder
        cross_encoder = CrossEncoderReranker()
        
        # Stage 2: Accurate LLM reranker (only on top 10 from stage 1)
        llm_reranker = LLMReranker(max_chunks=10)
        
        # Combined pipeline
        multi_stage = MultiStageReranker([
            (cross_encoder, 20),  # First stage: rerank top 20
            (llm_reranker, 10),   # Second stage: rerank top 10
        ])
        
        results = await multi_stage.rerank(query, chunks, top_k=5)
        ```
    """
    
    def __init__(
        self,
        stages: List[Tuple[Reranker, int]],
    ) -> None:
        """Initialize multi-stage reranker.
        
        Args:
            stages: List of (reranker, top_k) tuples, applied in order
        """
        self.stages = stages
        
        logger.info(
            "multi_stage_reranker_initialized",
            num_stages=len(stages),
            stage_names=[s[0].name for s in stages],
        )
    
    async def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Apply multi-stage reranking.
        
        Args:
            query: Search query
            chunks: List of chunk dictionaries
            top_k: Final number of results to return
            
        Returns:
            Reranked chunks after all stages
        """
        current_chunks = chunks.copy()
        
        for stage_idx, (reranker, stage_top_k) in enumerate(self.stages):
            if not current_chunks:
                break
            
            logger.info(
                "reranker_stage_start",
                stage=stage_idx + 1,
                reranker=reranker.name,
                chunks=len(current_chunks),
            )
            
            # Apply this stage
            current_chunks = await reranker.rerank(
                query,
                current_chunks,
                top_k=stage_top_k,
            )
            
            logger.info(
                "reranker_stage_complete",
                stage=stage_idx + 1,
                reranker=reranker.name,
                results=len(current_chunks),
            )
        
        # Apply final top_k
        if top_k and current_chunks:
            current_chunks = current_chunks[:top_k]
        
        return current_chunks
    
    @property
    def name(self) -> str:
        """Reranker name."""
        return f"multi-stage-{len(self.stages)}"


class RerankerPipeline:
    """Pipeline for applying reranking to search results.
    
    Provides a convenient interface for adding reranking to any search flow.
    
    Example:
        ```python
        pipeline = RerankerPipeline(
            reranker=CrossEncoderReranker(),
            rerank_top_k=20,
            final_top_k=10,
        )
        
        # Use with any search results
        final_results = await pipeline.rerank(query, search_results)
        ```
    """
    
    def __init__(
        self,
        reranker: Reranker,
        rerank_top_k: int = 20,
        final_top_k: int = 10,
        enable_debug: bool = False,
    ) -> None:
        """Initialize reranker pipeline.
        
        Args:
            reranker: Reranker instance to use
            rerank_top_k: Number of chunks to rerank
            final_top_k: Number of final results to return
            enable_debug: Enable debug logging
        """
        self.reranker = reranker
        self.rerank_top_k = rerank_top_k
        self.final_top_k = final_top_k
        self.enable_debug = enable_debug
        
        logger.info(
            "reranker_pipeline_initialized",
            reranker=reranker.name,
            rerank_top_k=rerank_top_k,
            final_top_k=final_top_k,
        )
    
    async def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Apply reranking pipeline to chunks.
        
        Args:
            query: Search query
            chunks: Search results to rerank
            
        Returns:
            Reranked results
        """
        if not chunks:
            return []
        
        # Limit to rerank_top_k
        chunks_to_rerank = chunks[:self.rerank_top_k]
        
        if self.enable_debug:
            logger.info(
                "reranker_pipeline_start",
                query=query[:50],
                input_chunks=len(chunks),
                rerank_chunks=len(chunks_to_rerank),
            )
        
        # Apply reranker
        reranked = await self.reranker.rerank(
            query,
            chunks_to_rerank,
            top_k=self.final_top_k,
        )
        
        if self.enable_debug:
            logger.info(
                "reranker_pipeline_complete",
                output_chunks=len(reranked),
                reranker=self.reranker.name,
            )
        
        return reranked
    
    async def rerank_with_fallback(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        fallback_reranker: Optional[Reranker] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank with fallback on failure.
        
        Args:
            query: Search query
            chunks: Search results
            fallback_reranker: Optional fallback reranker
            
        Returns:
            Reranked results (or original if both fail)
        """
        try:
            return await self.rerank(query, chunks)
        except Exception as e:
            logger.error(
                "primary_reranker_failed",
                reranker=self.reranker.name,
                error=str(e),
            )
            
            if fallback_reranker:
                try:
                    logger.info("attempting_fallback_reranker", fallback=fallback_reranker.name)
                    
                    fallback_pipeline = RerankerPipeline(
                        reranker=fallback_reranker,
                        rerank_top_k=self.rerank_top_k,
                        final_top_k=self.final_top_k,
                    )
                    return await fallback_pipeline.rerank(query, chunks)
                except Exception as fallback_e:
                    logger.error(
                        "fallback_reranker_failed",
                        error=str(fallback_e),
                    )
            
            # Return original chunks if all reranking fails
            return chunks[:self.final_top_k]
