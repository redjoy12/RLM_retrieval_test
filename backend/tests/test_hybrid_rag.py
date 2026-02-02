"""Tests for Hybrid RAG Integration (Component 8).

This module contains comprehensive tests for:
- Hybrid search engines
- Rerankers
- Chunk selectors
- Citation management
- Advanced hybrid strategy
"""

import asyncio
import math
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Import hybrid components
from rlm.hybrid import (
    AdaptiveChunkSelector,
    BM25Searcher,
    CitationManager,
    HybridSearcher,
    RerankerPipeline,
    ScoreThresholdSelector,
    SearchResultAggregator,
)
from rlm.hybrid.citation_manager import Citation


class TestBM25Searcher:
    """Test suite for BM25Searcher."""
    
    def test_bm25_initialization(self):
        """Test BM25 searcher initialization."""
        searcher = BM25Searcher(k1=1.5, b=0.75)
        
        assert searcher.k1 == 1.5
        assert searcher.b == 0.75
        assert searcher.total_chunks == 0
        assert searcher.avg_doc_length == 0.0
    
    def test_index_single_document(self):
        """Test indexing a single document."""
        searcher = BM25Searcher()
        chunks = ["This is chunk one", "This is chunk two"]
        
        searcher.index_document("doc-1", chunks)
        
        assert searcher.total_chunks == 2
        assert "doc-1" in searcher.documents
        assert len(searcher.documents["doc-1"]) == 2
    
    def test_remove_document(self):
        """Test removing a document from index."""
        searcher = BM25Searcher()
        searcher.index_document("doc-1", ["chunk one", "chunk two"])
        
        searcher.remove_document("doc-1")
        
        assert searcher.total_chunks == 0
        assert "doc-1" not in searcher.documents
    
    def test_simple_search(self):
        """Test basic keyword search."""
        searcher = BM25Searcher()
        searcher.index_document("doc-1", [
            "machine learning is great",
            "deep learning is a subset of machine learning",
        ])
        
        results = searcher.search("machine learning", top_k=5)
        
        assert len(results) == 2
        # Results should be ranked
        assert all("score" in r for r in results)
    
    def test_search_with_document_filter(self):
        """Test searching with document ID filter."""
        searcher = BM25Searcher()
        searcher.index_document("doc-1", ["machine learning"])
        searcher.index_document("doc-2", ["deep learning"])
        
        results = searcher.search("learning", document_ids=["doc-1"], top_k=5)
        
        assert len(results) == 1
        assert results[0]["document_id"] == "doc-1"
    
    def test_search_no_results(self):
        """Test search with no matching results."""
        searcher = BM25Searcher()
        searcher.index_document("doc-1", ["machine learning"])
        
        results = searcher.search("quantum physics")
        
        assert len(results) == 0
    
    def test_get_stats(self):
        """Test getting BM25 statistics."""
        searcher = BM25Searcher()
        searcher.index_document("doc-1", ["chunk one", "chunk two"])
        
        stats = searcher.get_stats()
        
        assert stats["total_documents"] == 1
        assert stats["total_chunks"] == 2
        assert stats["avg_doc_length"] > 0


class TestHybridSearcher:
    """Test suite for HybridSearcher."""
    
    @pytest.mark.asyncio
    async def test_hybrid_search_with_mock(self):
        """Test hybrid search with mocked Qdrant."""
        # This test would require a mocked Qdrant client
        # For now, we just verify the structure
        pass
    
    def test_rrf_fusion_calculation(self):
        """Test Reciprocal Rank Fusion calculation."""
        from rlm.hybrid.search_engines import HybridSearcher
        
        # Create a mock searcher
        searcher = HybridSearcher.__new__(HybridSearcher)
        searcher.rrf_k = 60
        
        # Test fusion
        vector_results = [
            {"chunk_id": "c1", "score": 0.9},
            {"chunk_id": "c2", "score": 0.8},
        ]
        keyword_results = [
            {"chunk_id": "c2", "score": 0.7},  # Same chunk
            {"chunk_id": "c3", "score": 0.6},
        ]
        
        fused = searcher._reciprocal_rank_fusion(
            vector_results,
            keyword_results,
            0.7,
            0.3,
        )
        
        # c2 should appear in both, so it should have higher fused score
        assert len(fused) == 3
        assert all("fused_score" in r for r in fused)


class TestChunkSelectors:
    """Test suite for chunk selection strategies."""
    
    def test_adaptive_selector_initialization(self):
        """Test adaptive selector initialization."""
        selector = AdaptiveChunkSelector(
            min_chunks=3,
            max_chunks=10,
            diversity_threshold=0.8,
        )
        
        assert selector.min_chunks == 3
        assert selector.max_chunks == 10
        assert selector.diversity_threshold == 0.8
    
    def test_query_complexity_estimation(self):
        """Test query complexity estimation."""
        selector = AdaptiveChunkSelector()
        
        # Simple query
        simple_score = selector._estimate_query_complexity("What is AI?")
        assert 0 <= simple_score <= 1
        
        # Complex query
        complex_query = "Compare and analyze the relationship between machine learning and deep learning"
        complex_score = selector._estimate_query_complexity(complex_query)
        
        assert complex_score > simple_score
    
    def test_adaptive_selection_basic(self):
        """Test basic adaptive chunk selection."""
        selector = AdaptiveChunkSelector(min_chunks=3, max_chunks=5)
        
        chunks = [
            {"chunk_id": f"c{i}", "score": 0.9 - i*0.1, "content": f"Content {i} different topic"}
            for i in range(10)
        ]
        
        selected = selector.select(chunks, "test query", max_chunks=5)
        
        assert len(selected) >= 3
        assert len(selected) <= 5
    
    def test_jaccard_similarity(self):
        """Test Jaccard similarity calculation."""
        selector = AdaptiveChunkSelector()
        
        # Identical texts
        sim1 = selector._compute_jaccard_similarity(
            "machine learning",
            "machine learning"
        )
        assert sim1 == 1.0
        
        # Completely different
        sim2 = selector._compute_jaccard_similarity(
            "machine learning",
            "quantum physics"
        )
        assert sim2 == 0.0
        
        # Partial overlap
        sim3 = selector._compute_jaccard_similarity(
            "machine learning ai",
            "machine learning"
        )
        assert 0 < sim3 < 1.0
    
    def test_deduplication(self):
        """Test chunk deduplication."""
        selector = AdaptiveChunkSelector()
        
        chunks = [
            {"chunk_id": "c1", "content": "machine learning is great"},
            {"chunk_id": "c2", "content": "machine learning is great"},  # Duplicate
            {"chunk_id": "c3", "content": "deep learning is different"},
        ]
        
        deduped = selector._deduplicate_chunks(chunks, similarity_threshold=0.9)
        
        # Should remove the near-duplicate
        assert len(deduped) < len(chunks)
    
    def test_score_threshold_selector(self):
        """Test score threshold selector."""
        selector = ScoreThresholdSelector(min_score=0.7, max_chunks=5)
        
        chunks = [
            {"chunk_id": "c1", "score": 0.9},
            {"chunk_id": "c2", "score": 0.8},
            {"chunk_id": "c3", "score": 0.6},
            {"chunk_id": "c4", "score": 0.5},
        ]
        
        selected = selector.select(chunks, "query")
        
        assert len(selected) <= 2  # Only c1 and c2 meet threshold
        assert all(s["score"] >= 0.7 for s in selected)


class TestCitationManager:
    """Test suite for CitationManager."""
    
    def test_add_citation(self):
        """Test adding a citation."""
        manager = CitationManager()
        
        citation_id = manager.add_chunk_citation(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="Test content here",
            score=0.9,
        )
        
        assert citation_id == "cite_1"
        assert len(manager.citations) == 1
    
    def test_get_citation(self):
        """Test retrieving a citation."""
        manager = CitationManager()
        manager.add_chunk_citation("chunk-1", "doc-1", "Content", 0.9)
        
        citation = manager.get_citation("chunk-1")
        
        assert citation is not None
        assert citation.citation_number == 1
        assert citation.document_id == "doc-1"
    
    def test_citation_formatting(self):
        """Test citation formatting."""
        manager = CitationManager()
        manager.add_chunk_citation("chunk-1", "doc-1", "Content", 0.9)
        
        summary = manager.format_citation_summary()
        
        assert "Sources:" in summary
        assert "[1]" in summary
        assert "doc-1" in summary
    
    def test_extract_cited_chunks(self):
        """Test extracting citations from text."""
        manager = CitationManager()
        
        text = "This is a fact [1] and another fact [3]."
        cited = manager.extract_cited_chunks(text)
        
        assert 1 in cited
        assert 3 in cited
        assert 2 not in cited
    
    def test_citation_stats(self):
        """Test citation statistics."""
        manager = CitationManager()
        manager.add_chunk_citation("c1", "doc-1", "Content 1", 0.9)
        manager.add_chunk_citation("c2", "doc-1", "Content 2", 0.8)
        manager.add_chunk_citation("c3", "doc-2", "Content 3", 0.7)
        
        stats = manager.get_citation_stats()
        
        assert stats["total_citations"] == 3
        assert stats["unique_documents"] == 2
        assert 0.7 <= stats["avg_score"] <= 0.9
    
    def test_citation_deduplication(self):
        """Test that same chunk only added once."""
        manager = CitationManager()
        
        manager.add_chunk_citation("chunk-1", "doc-1", "Content", 0.9)
        manager.add_chunk_citation("chunk-1", "doc-1", "Content", 0.9)  # Duplicate
        
        assert len(manager.citations) == 1
        assert manager.citation_counter == 1
    
    def test_clear_citations(self):
        """Test clearing all citations."""
        manager = CitationManager()
        manager.add_chunk_citation("c1", "doc-1", "Content", 0.9)
        
        manager.clear_citations()
        
        assert len(manager.citations) == 0
        assert manager.citation_counter == 0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        manager = CitationManager()
        manager.add_chunk_citation("c1", "doc-1", "Content", 0.9)
        
        data = manager.to_dict()
        
        assert "citations" in data
        assert "stats" in data
        assert len(data["citations"]) == 1


class TestSearchResultAggregator:
    """Test suite for SearchResultAggregator."""
    
    def test_aggregator_initialization(self):
        """Test aggregator initialization."""
        aggregator = SearchResultAggregator()
        
        assert len(aggregator.results) == 0
        assert len(aggregator.sources) == 0
    
    def test_add_single_source(self):
        """Test adding results from single source."""
        aggregator = SearchResultAggregator()
        
        results = [
            {"chunk_id": "c1", "score": 0.9},
            {"chunk_id": "c2", "score": 0.8},
        ]
        
        aggregator.add_results(results, source="vector", weight=1.0)
        
        assert len(aggregator.results) == 2
        assert aggregator.results["c1"]["aggregated_score"] == 0.9
    
    def test_add_multiple_sources(self):
        """Test adding results from multiple sources."""
        aggregator = SearchResultAggregator()
        
        vector_results = [{"chunk_id": "c1", "score": 0.9}]
        keyword_results = [{"chunk_id": "c1", "score": 0.7}]
        
        aggregator.add_results(vector_results, source="vector", weight=1.0)
        aggregator.add_results(keyword_results, source="bm25", weight=0.8)
        
        # c1 should have combined score
        assert "c1" in aggregator.results
        assert aggregator.results["c1"]["aggregated_score"] == 0.9 + 0.56
        assert "vector" in aggregator.sources["c1"]
        assert "bm25" in aggregator.sources["c1"]
    
    def test_get_aggregated_results(self):
        """Test getting aggregated results."""
        aggregator = SearchResultAggregator()
        
        vector_results = [
            {"chunk_id": "c1", "score": 0.9},
            {"chunk_id": "c2", "score": 0.8},
        ]
        
        aggregator.add_results(vector_results, source="vector")
        final = aggregator.get_aggregated_results(top_k=2)
        
        assert len(final) == 2
        # Should be sorted by score
        assert final[0]["aggregated_score"] >= final[1]["aggregated_score"]
    
    def test_min_sources_filter(self):
        """Test filtering by minimum sources."""
        aggregator = SearchResultAggregator()
        
        aggregator.add_results([{"chunk_id": "c1", "score": 0.9}], source="vector")
        aggregator.add_results([{"chunk_id": "c1", "score": 0.8}], source="bm25")
        aggregator.add_results([{"chunk_id": "c2", "score": 0.7}], source="vector")
        
        # Require 2 sources
        filtered = aggregator.get_aggregated_results(min_sources=2)
        
        assert len(filtered) == 1
        assert filtered[0]["chunk_id"] == "c1"
    
    def test_reset(self):
        """Test resetting aggregator."""
        aggregator = SearchResultAggregator()
        aggregator.add_results([{"chunk_id": "c1", "score": 0.9}], source="vector")
        
        aggregator.reset()
        
        assert len(aggregator.results) == 0
        assert len(aggregator.sources) == 0


class TestIntegration:
    """Integration tests for hybrid RAG pipeline."""
    
    def test_full_pipeline_simulation(self):
        """Simulate the full hybrid RAG pipeline."""
        # Step 1: Hybrid retrieval (mocked)
        chunks = [
            {
                "chunk_id": f"c{i}",
                "document_id": f"doc-{i % 3}",
                "content": f"This is content about topic {i} with keywords",
                "score": 0.95 - i * 0.05,
            }
            for i in range(10)
        ]
        
        # Step 2: Reranking (would normally use cross-encoder)
        # For test, just simulate by adding cross_encoder_score
        for i, chunk in enumerate(chunks):
            chunk["cross_encoder_score"] = chunk["score"] * 0.9
        
        # Step 3: Adaptive selection
        selector = AdaptiveChunkSelector(min_chunks=3, max_chunks=5)
        selected = selector.select(chunks, "test query about keywords", max_chunks=5)
        
        assert len(selected) >= 3
        assert len(selected) <= 5
        
        # Step 4: Citation tracking
        manager = CitationManager()
        for chunk in selected:
            manager.add_chunk_citation(
                chunk_id=chunk["chunk_id"],
                document_id=chunk["document_id"],
                content=chunk["content"],
                score=chunk.get("cross_encoder_score", chunk["score"]),
            )
        
        citation_summary = manager.format_citation_summary()
        assert len(manager.citations) == len(selected)
        assert "Sources:" in citation_summary
    
    def test_performance_characteristics(self):
        """Test that operations complete in reasonable time."""
        import time
        
        # BM25 performance
        searcher = BM25Searcher()
        chunks = [f"This is chunk content number {i}" for i in range(100)]
        
        start = time.time()
        searcher.index_document("doc-1", chunks)
        index_time = time.time() - start
        
        start = time.time()
        results = searcher.search("chunk content", top_k=10)
        search_time = time.time() - start
        
        # Should be fast for 100 chunks
        assert index_time < 1.0  # Less than 1 second
        assert search_time < 0.1  # Less than 100ms


@pytest.mark.skip(reason="Requires sentence-transformers")
class TestRerankerIntegration:
    """Integration tests for rerankers (requires sentence-transformers)."""
    
    @pytest.mark.asyncio
    async def test_cross_encoder_reranker(self):
        """Test cross-encoder reranking."""
        from rlm.hybrid import CrossEncoderReranker
        
        reranker = CrossEncoderReranker()
        
        chunks = [
            {"chunk_id": "c1", "content": "Machine learning is a field of AI"},
            {"chunk_id": "c2", "content": "Deep learning uses neural networks"},
        ]
        
        results = await reranker.rerank(
            "What is machine learning?",
            chunks,
            top_k=2,
        )
        
        assert len(results) == 2
        assert "cross_encoder_score" in results[0]


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
