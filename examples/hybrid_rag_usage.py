"""Hybrid RAG Usage Examples - Component 8.

This file demonstrates how to use the advanced hybrid RAG features.

Requirements:
    pip install rank-bm25 sentence-transformers
    # Ensure Qdrant is running for full functionality

"""

import asyncio
from typing import Any, Dict, List


async def example_1_basic_hybrid_search():
    """Example 1: Basic Hybrid Search (Semantic + Keyword)."""
    print("=" * 60)
    print("Example 1: Basic Hybrid Search")
    print("=" * 60)
    
    from rlm.hybrid import HybridSearcher
    
    # Initialize hybrid searcher
    # Note: Requires Qdrant to be running
    searcher = HybridSearcher()
    
    # Search query
    query = "What are the revenue trends in Q3?"
    document_ids = ["doc-1", "doc-2"]  # Your document IDs
    
    # Perform hybrid search
    results = await searcher.search(
        query=query,
        document_ids=document_ids,
        top_k=10,
        semantic_weight=0.7,  # 70% weight to semantic search
        keyword_weight=0.3,   # 30% weight to keyword search
    )
    
    print(f"\nQuery: {query}")
    print(f"Results: {len(results)}")
    
    for i, result in enumerate(results[:5]):
        print(f"\n{i+1}. Score: {result.get('fused_score', result.get('score', 0)):.3f}")
        print(f"   Content: {result['content'][:100]}...")


async def example_2_reranking():
    """Example 2: Cross-Encoder Reranking."""
    print("\n" + "=" * 60)
    print("Example 2: Cross-Encoder Reranking")
    print("=" * 60)
    
    from rlm.hybrid import RerankerPipeline, CrossEncoderReranker
    
    # Initialize reranker
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu",  # or "cuda" if available
    )
    
    # Create pipeline
    pipeline = RerankerPipeline(
        reranker=reranker,
        rerank_top_k=20,  # Rerank top 20 from initial retrieval
        final_top_k=10,   # Return top 10 after reranking
    )
    
    # Mock search results (normally from hybrid search)
    search_results = [
        {"chunk_id": f"c{i}", "content": f"Relevant content about topic {i}", "score": 0.9 - i*0.05}
        for i in range(20)
    ]
    
    query = "What is machine learning?"
    
    # Apply reranking
    reranked_results = await pipeline.rerank(query, search_results)
    
    print(f"\nQuery: {query}")
    print(f"Input chunks: {len(search_results)}")
    print(f"Output chunks: {len(reranked_results)}")
    
    for i, result in enumerate(reranked_results[:3]):
        print(f"\n{i+1}. Cross-Encoder Score: {result.get('cross_encoder_score', 0):.3f}")
        print(f"   Content: {result['content'][:80]}...")


async def example_3_adaptive_selection():
    """Example 3: Adaptive Chunk Selection."""
    print("\n" + "=" * 60)
    print("Example 3: Adaptive Chunk Selection")
    print("=" * 60)
    
    from rlm.hybrid import AdaptiveChunkSelector
    
    # Initialize selector
    selector = AdaptiveChunkSelector(
        min_chunks=3,
        max_chunks=10,
        diversity_threshold=0.8,  # Avoid redundant content
    )
    
    # Mock chunks with varying content
    chunks = [
        {
            "chunk_id": "c1",
            "score": 0.95,
            "content": "Machine learning is a subset of AI focused on algorithms that learn from data",
        },
        {
            "chunk_id": "c2", 
            "score": 0.90,
            "content": "Deep learning uses neural networks with multiple layers",
        },
        {
            "chunk_id": "c3",
            "score": 0.88,
            "content": "Supervised learning requires labeled training data",
        },
        {
            "chunk_id": "c4",
            "score": 0.85,
            "content": "Unsupervised learning finds patterns without labels",
        },
        {
            "chunk_id": "c5",
            "score": 0.80,
            "content": "Reinforcement learning uses rewards to train agents",
        },
        {
            "chunk_id": "c6",
            "score": 0.75,
            "content": "Machine learning is a subset of AI focused on algorithms",  # Near-duplicate of c1
        },
    ]
    
    # Simple query - should select fewer chunks
    simple_query = "What is AI?"
    simple_selected = selector.select(chunks, simple_query, max_chunks=10)
    
    # Complex query - should select more chunks
    complex_query = "Compare and analyze different types of machine learning approaches including supervised, unsupervised, and reinforcement learning"
    complex_selected = selector.select(chunks, complex_query, max_chunks=10)
    
    print(f"\nSimple Query: '{simple_query}'")
    print(f"Chunks Selected: {len(simple_selected)}")
    
    print(f"\nComplex Query: '{complex_query}'")
    print(f"Chunks Selected: {len(complex_selected)}")
    
    print(f"\nComplex query selected more chunks: {len(complex_selected) >= len(simple_selected)}")


async def example_4_citations():
    """Example 4: Citation Tracking."""
    print("\n" + "=" * 60)
    print("Example 4: Citation Tracking")
    print("=" * 60)
    
    from rlm.hybrid import CitationManager
    
    # Initialize citation manager
    manager = CitationManager()
    
    # Add chunks as citations
    selected_chunks = [
        {
            "chunk_id": "chunk-1",
            "document_id": "annual-report-2024.pdf",
            "content": "Revenue increased by 15% in Q3 2024 compared to Q2.",
            "score": 0.92,
        },
        {
            "chunk_id": "chunk-2",
            "document_id": "annual-report-2024.pdf",
            "content": "Operating costs decreased by 8% due to efficiency improvements.",
            "score": 0.88,
        },
        {
            "chunk_id": "chunk-3",
            "document_id": "financial-summary.pdf",
            "content": "Net profit margin improved to 12.5% in Q3.",
            "score": 0.85,
        },
    ]
    
    for chunk in selected_chunks:
        manager.add_chunk_citation(
            chunk_id=chunk["chunk_id"],
            document_id=chunk["document_id"],
            content=chunk["content"],
            score=chunk["score"],
        )
    
    # Generate citation summary
    citation_summary = manager.format_citation_summary(
        format_type="numbered",
        include_scores=True,
    )
    
    print("\nGenerated Citations:")
    print(citation_summary)
    
    # Get citation statistics
    stats = manager.get_citation_stats()
    print(f"\nCitation Statistics:")
    print(f"  Total citations: {stats['total_citations']}")
    print(f"  Unique documents: {stats['unique_documents']}")
    print(f"  Average score: {stats['avg_score']:.3f}")


async def example_5_advanced_hybrid_strategy():
    """Example 5: Full Advanced Hybrid Strategy."""
    print("\n" + "=" * 60)
    print("Example 5: Advanced Hybrid Strategy (Full Pipeline)")
    print("=" * 60)
    
    from rlm.hybrid import AdvancedHybridStrategy
    from rlm.routing.models import CostEstimate
    
    # Initialize strategy
    strategy = AdvancedHybridStrategy()
    
    # Setup cost estimate
    cost_estimate = CostEstimate(
        estimated_input_tokens=8000,
        estimated_output_tokens=1000,
        estimated_total_tokens=9000,
        estimated_cost_usd=0.015,
        model_used="gpt-4o",
    )
    
    # Execute strategy
    query = "What are the key financial trends and performance metrics?"
    document_ids = ["doc-1", "doc-2", "doc-3"]
    
    print(f"\nQuery: {query}")
    print(f"Documents: {len(document_ids)}")
    print("\nExecuting hybrid strategy with:")
    print("  1. Hybrid retrieval (semantic + keyword)")
    print("  2. Cross-encoder reranking")
    print("  3. Adaptive chunk selection")
    print("  4. RLM deep analysis")
    print("  5. Citation tracking")
    
    # Note: This would actually execute if documents exist
    # result = await strategy.execute(
    #     query=query,
    #     document_ids=document_ids,
    #     cost_estimate=cost_estimate,
    #     enable_reranking=True,
    #     enable_citations=True,
    #     enable_adaptive=True,
    # )
    
    print("\n[Execution would happen here with actual documents]")


async def example_6_bm25_only():
    """Example 6: Using BM25 Keyword Search Only."""
    print("\n" + "=" * 60)
    print("Example 6: BM25 Keyword Search Only")
    print("=" * 60)
    
    from rlm.hybrid import BM25Searcher
    
    # Initialize BM25 searcher
    searcher = BM25Searcher(k1=1.5, b=0.75)
    
    # Index some documents
    documents = {
        "doc-1": [
            "Machine learning is a method of data analysis",
            "ML algorithms build models from sample data",
            "Deep learning is part of machine learning",
        ],
        "doc-2": [
            "Quantum computing uses quantum-mechanical phenomena",
            "QC is different from classical computing",
            "Quantum bits are the basic unit",
        ],
    }
    
    for doc_id, chunks in documents.items():
        searcher.index_document(doc_id, chunks)
    
    # Search with BM25
    query = "machine learning algorithms"
    results = searcher.search(query, top_k=5)
    
    print(f"\nQuery: {query}")
    print(f"Results: {len(results)}")
    
    for i, result in enumerate(results):
        print(f"\n{i+1}. Doc: {result['document_id']}, Score: {result['score']:.3f}")
        print(f"   {result['content'][:60]}...")
    
    # Show statistics
    stats = searcher.get_stats()
    print(f"\nIndex Statistics:")
    print(f"  Documents: {stats['total_documents']}")
    print(f"  Chunks: {stats['total_chunks']}")
    print(f"  Unique terms: {stats['unique_terms']}")


async def example_7_multi_stage_reranking():
    """Example 7: Multi-Stage Reranking."""
    print("\n" + "=" * 60)
    print("Example 7: Multi-Stage Reranking")
    print("=" * 60)
    
    print("\nMulti-stage reranking uses:")
    print("  Stage 1: Cross-encoder (fast, filters to top 20)")
    print("  Stage 2: LLM reranker (accurate, selects top 10)")
    
    print("\n[Requires sentence-transformers and LLM client]")
    print("Example code:")
    print("""
    from rlm.hybrid import MultiStageReranker, CrossEncoderReranker, LLMReranker
    
    # Create multi-stage pipeline
    cross_encoder = CrossEncoderReranker()
    llm_reranker = LLMReranker(model="gpt-5-mini", max_chunks=10)
    
    multi_stage = MultiStageReranker([
        (cross_encoder, 20),  # First stage
        (llm_reranker, 10),   # Second stage
    ])
    
    results = await multi_stage.rerank(query, chunks)
    """)


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Hybrid RAG Integration - Usage Examples")
    print("=" * 60)
    
    try:
        await example_1_basic_hybrid_search()
    except Exception as e:
        print(f"\nExample 1 skipped (requires Qdrant): {e}")
    
    try:
        await example_2_reranking()
    except Exception as e:
        print(f"\nExample 2 skipped (requires sentence-transformers): {e}")
    
    await example_3_adaptive_selection()
    await example_4_citations()
    await example_5_advanced_hybrid_strategy()
    await example_6_bm25_only()
    await example_7_multi_stage_reranking()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
