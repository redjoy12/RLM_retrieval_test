"""Hybrid search engines combining semantic and keyword search.

This module provides advanced search capabilities that combine vector-based
semantic search with BM25 keyword search for optimal retrieval performance.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import structlog
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

logger = structlog.get_logger()


class BM25Searcher:
    """BM25 keyword-based search engine.
    
    Implements the BM25 ranking function for keyword search over document chunks.
    This provides complementary retrieval to semantic search for keyword-heavy queries.
    
    Example:
        ```python
        searcher = BM25Searcher()
        searcher.index_document("doc-1", ["chunk1 text", "chunk2 text"])
        results = searcher.search("specific keyword", top_k=5)
        ```
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Initialize BM25 searcher.
        
        Args:
            k1: Term frequency saturation parameter (default 1.5)
            b: Length normalization parameter (default 0.75)
        """
        self.k1 = k1
        self.b = b
        self.documents: Dict[str, List[str]] = {}  # doc_id -> list of chunks
        self.chunk_index: Dict[str, Tuple[str, int]] = {}  # chunk_id -> (doc_id, index)
        self.inverted_index: Dict[str, Dict[str, int]] = {}  # term -> {chunk_id: freq}
        self.avg_doc_length = 0.0
        self.total_chunks = 0
        
        logger.info("bm25_searcher_initialized", k1=k1, b=b)
    
    def index_document(self, document_id: str, chunks: List[str]) -> None:
        """Index a document's chunks for BM25 search.
        
        Args:
            document_id: Document identifier
            chunks: List of text chunks from the document
        """
        if document_id in self.documents:
            # Remove existing document before re-indexing
            self.remove_document(document_id)
        
        self.documents[document_id] = chunks
        
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{idx}"
            self.chunk_index[chunk_id] = (document_id, idx)
            
            # Tokenize and build inverted index
            terms = self._tokenize(chunk)
            term_freq = {}
            for term in terms:
                term_freq[term] = term_freq.get(term, 0) + 1
            
            for term, freq in term_freq.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                self.inverted_index[term][chunk_id] = freq
        
        self.total_chunks += len(chunks)
        self._update_avg_doc_length()
        
        logger.info(
            "bm25_document_indexed",
            document_id=document_id,
            chunks_added=len(chunks),
        )
    
    def remove_document(self, document_id: str) -> None:
        """Remove a document from the BM25 index.
        
        Args:
            document_id: Document to remove
        """
        if document_id not in self.documents:
            return
        
        chunks = self.documents[document_id]
        
        for idx in range(len(chunks)):
            chunk_id = f"{document_id}_chunk_{idx}"
            
            if chunk_id in self.chunk_index:
                del self.chunk_index[chunk_id]
            
            # Remove from inverted index
            for term in list(self.inverted_index.keys()):
                if chunk_id in self.inverted_index[term]:
                    del self.inverted_index[term][chunk_id]
                    if not self.inverted_index[term]:
                        del self.inverted_index[term]
        
        del self.documents[document_id]
        self.total_chunks -= len(chunks)
        self._update_avg_doc_length()
        
        logger.info("bm25_document_removed", document_id=document_id)
    
    def search(
        self, 
        query: str, 
        document_ids: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for chunks using BM25 scoring.
        
        Args:
            query: Search query
            document_ids: Optional filter to specific documents
            top_k: Number of results to return
            
        Returns:
            List of scored results with chunk_id, document_id, score, content
        """
        query_terms = self._tokenize(query)
        if not query_terms:
            return []
        
        scores: Dict[str, float] = {}
        
        for term in query_terms:
            if term not in self.inverted_index:
                continue
            
            idf = self._compute_idf(term)
            
            for chunk_id, freq in self.inverted_index[term].items():
                # Filter by document if specified
                if document_ids:
                    doc_id = self.chunk_index.get(chunk_id, (None, 0))[0]
                    if doc_id not in document_ids:
                        continue
                
                # Compute BM25 score for this term
                doc_id, _ = self.chunk_index[chunk_id]
                doc_length = len(self._tokenize(self._get_chunk_content(chunk_id)))
                
                score = idf * (
                    (freq * (self.k1 + 1)) / 
                    (freq + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length))
                )
                
                scores[chunk_id] = scores.get(chunk_id, 0) + score
        
        # Sort by score and return top_k
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for chunk_id, score in sorted_results[:top_k]:
            doc_id, idx = self.chunk_index[chunk_id]
            content = self._get_chunk_content(chunk_id)
            
            results.append({
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "chunk_index": idx,
                "content": content,
                "score": score,
                "search_type": "bm25",
            })
        
        logger.info(
            "bm25_search_complete",
            query=query[:50],
            results_found=len(results),
            top_score=results[0]["score"] if results else 0,
        )
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of lowercase tokens
        """
        # Simple tokenization: lowercase, split on whitespace, remove punctuation
        text = text.lower()
        tokens = []
        for word in text.split():
            # Remove punctuation
            word = ''.join(c for c in word if c.isalnum())
            if word and len(word) > 2:  # Filter very short tokens
                tokens.append(word)
        return tokens
    
    def _compute_idf(self, term: str) -> float:
        """Compute IDF for a term.
        
        Args:
            term: Term to compute IDF for
            
        Returns:
            IDF score
        """
        doc_freq = len(self.inverted_index.get(term, {}))
        if doc_freq == 0:
            return 0
        return math.log((self.total_chunks - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
    
    def _update_avg_doc_length(self) -> None:
        """Update average document length."""
        if self.total_chunks == 0:
            self.avg_doc_length = 0
            return
        
        total_length = sum(
            len(self._tokenize(chunk))
            for chunks in self.documents.values()
            for chunk in chunks
        )
        self.avg_doc_length = total_length / self.total_chunks
    
    def _get_chunk_content(self, chunk_id: str) -> str:
        """Get content for a chunk_id."""
        doc_id, idx = self.chunk_index[chunk_id]
        return self.documents[doc_id][idx]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the BM25 index.
        
        Returns:
            Dictionary with index statistics
        """
        return {
            "total_documents": len(self.documents),
            "total_chunks": self.total_chunks,
            "unique_terms": len(self.inverted_index),
            "avg_doc_length": self.avg_doc_length,
        }


class HybridSearcher:
    """Hybrid search combining vector (semantic) and BM25 (keyword) search.
    
    Uses Reciprocal Rank Fusion (RRF) to combine results from both search methods,
    providing optimal retrieval for both semantic and keyword-heavy queries.
    
    Example:
        ```python
        searcher = HybridSearcher(qdrant_client)
        
        # Index documents
        searcher.index_document("doc-1", ["chunk1", "chunk2"], embeddings)
        
        # Hybrid search
        results = searcher.search(
            "What is machine learning?",
            document_ids=["doc-1"],
            top_k=10,
            semantic_weight=0.7,
            keyword_weight=0.3,
        )
        ```
    """
    
    def __init__(
        self,
        qdrant_client: Optional[QdrantClient] = None,
        collection_name: str = "rlm_chunks",
        embedding_model: str = "text-embedding-3-small",
        rrf_k: int = 60,
    ) -> None:
        """Initialize hybrid searcher.
        
        Args:
            qdrant_client: Qdrant client for vector search
            collection_name: Name of Qdrant collection
            embedding_model: Model for generating embeddings
            rrf_k: RRF fusion parameter (default 60)
        """
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.rrf_k = rrf_k
        
        # Initialize BM25 searcher
        self.bm25 = BM25Searcher()
        
        # Track which documents are indexed in BM25
        self.bm25_indexed_docs: set = set()
        
        logger.info(
            "hybrid_searcher_initialized",
            collection=collection_name,
            rrf_k=rrf_k,
        )
    
    async def index_document(
        self, 
        document_id: str, 
        chunks: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        """Index a document for hybrid search.
        
        Args:
            document_id: Document identifier
            chunks: List of chunk dicts with 'content' and optionally 'index'
            embeddings: Optional pre-computed embeddings for vector search
        """
        # Index in BM25
        chunk_contents = [chunk.get("content", "") for chunk in chunks]
        self.bm25.index_document(document_id, chunk_contents)
        self.bm25_indexed_docs.add(document_id)
        
        logger.info(
            "hybrid_document_indexed",
            document_id=document_id,
            chunks=len(chunks),
        )
    
    def remove_document(self, document_id: str) -> None:
        """Remove a document from hybrid search index.
        
        Args:
            document_id: Document to remove
        """
        self.bm25.remove_document(document_id)
        self.bm25_indexed_docs.discard(document_id)
        
        logger.info("hybrid_document_removed", document_id=document_id)
    
    async def search(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        vector_top_k: int = 20,
        keyword_top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector and BM25 results.
        
        Args:
            query: Search query
            document_ids: Optional filter to specific documents
            top_k: Number of final results to return
            semantic_weight: Weight for vector search (0-1)
            keyword_weight: Weight for BM25 search (0-1)
            vector_top_k: Number of results from vector search
            keyword_top_k: Number of results from BM25 search
            
        Returns:
            Fused and ranked results
        """
        # Get results from both search methods
        vector_results = []
        keyword_results = []
        
        if semantic_weight > 0:
            vector_results = await self._vector_search(
                query, document_ids, vector_top_k
            )
        
        if keyword_weight > 0:
            keyword_results = self.bm25.search(query, document_ids, keyword_top_k)
        
        # Fuse results using weighted RRF
        fused = self._reciprocal_rank_fusion(
            vector_results, 
            keyword_results,
            semantic_weight,
            keyword_weight,
        )
        
        # Return top_k
        final_results = fused[:top_k]
        
        logger.info(
            "hybrid_search_complete",
            query=query[:50],
            vector_results=len(vector_results),
            keyword_results=len(keyword_results),
            final_results=len(final_results),
        )
        
        return final_results
    
    async def search_semantic_only(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search using only vector/semantic search.
        
        Args:
            query: Search query
            document_ids: Optional filter to specific documents
            top_k: Number of results
            
        Returns:
            Vector search results
        """
        results = await self._vector_search(query, document_ids, top_k)
        
        # Add search_type field
        for r in results:
            r["search_type"] = "semantic"
        
        return results
    
    def search_keyword_only(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search using only BM25 keyword search.
        
        Args:
            query: Search query
            document_ids: Optional filter to specific documents
            top_k: Number of results
            
        Returns:
            BM25 search results
        """
        return self.bm25.search(query, document_ids, top_k)
    
    async def _vector_search(
        self,
        query: str,
        document_ids: Optional[List[str]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Perform vector search using Qdrant.
        
        Args:
            query: Search query
            document_ids: Optional filter to specific documents
            top_k: Number of results
            
        Returns:
            Vector search results
        """
        # Generate embedding for query
        query_embedding = await self._generate_embedding(query)
        
        # Build filter
        query_filter = None
        if document_ids:
            from qdrant_client.models import FieldCondition, Filter, MatchValue
            should_conditions = [
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=doc_id),
                )
                for doc_id in document_ids
            ]
            query_filter = Filter(should=should_conditions)
        
        # Search Qdrant
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=top_k,
        )
        
        results = []
        for point in search_result:
            results.append({
                "chunk_id": point.id,
                "document_id": point.payload.get("document_id", "unknown"),
                "chunk_index": point.payload.get("chunk_index", 0),
                "content": point.payload.get("content", ""),
                "score": point.score,
                "search_type": "semantic",
            })
        
        return results
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        vector_weight: float,
        keyword_weight: float,
    ) -> List[Dict[str, Any]]:
        """Fuse results from vector and keyword search using weighted RRF.
        
        Args:
            vector_results: Results from vector search
            keyword_results: Results from BM25 search
            vector_weight: Weight for vector results (0-1)
            keyword_weight: Weight for keyword results (0-1)
            
        Returns:
            Fused and re-ranked results
        """
        scores: Dict[str, float] = {}
        result_map: Dict[str, Dict[str, Any]] = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results):
            chunk_id = result["chunk_id"]
            rrf_score = vector_weight * (1 / (self.rrf_k + rank + 1))
            scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score
            result_map[chunk_id] = result
        
        # Process keyword results
        for rank, result in enumerate(keyword_results):
            chunk_id = result["chunk_id"]
            rrf_score = keyword_weight * (1 / (self.rrf_k + rank + 1))
            scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score
            if chunk_id not in result_map:
                result_map[chunk_id] = result
        
        # Sort by fused score
        sorted_results = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        # Build final results
        final_results = []
        for chunk_id, fused_score in sorted_results:
            result = result_map[chunk_id].copy()
            result["fused_score"] = fused_score
            result["rrf_rank"] = len(final_results) + 1
            final_results.append(result)
        
        return final_results
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            import openai
            
            client = openai.AsyncOpenAI()
            
            response = await client.embeddings.create(
                model=self.embedding_model,
                input=text[:8000],  # Limit to avoid token limits
            )
            
            return response.data[0].embedding
        except Exception as e:
            logger.error(
                "embedding_generation_failed",
                text_length=len(text),
                error=str(e),
            )
            # Return zero vector as fallback
            return [0.0] * 1536
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the hybrid search index.
        
        Returns:
            Dictionary with index statistics
        """
        return {
            "bm25_stats": self.bm25.get_stats(),
            "bm25_indexed_documents": len(self.bm25_indexed_docs),
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model,
            "rrf_k": self.rrf_k,
        }


class SearchResultAggregator:
    """Aggregate and deduplicate search results from multiple sources.
    
    Useful for combining results from different search strategies.
    
    Example:
        ```python
        aggregator = SearchResultAggregator()
        
        # Add results from different searches
        aggregator.add_results(vector_results, source="vector", weight=1.0)
        aggregator.add_results(bm25_results, source="bm25", weight=0.8)
        
        # Get aggregated results
        final = aggregator.get_aggregated_results(top_k=10)
        ```
    """
    
    def __init__(self) -> None:
        """Initialize the aggregator."""
        self.results: Dict[str, Dict[str, Any]] = {}
        self.sources: Dict[str, List[str]] = {}  # chunk_id -> list of sources
    
    def add_results(
        self, 
        results: List[Dict[str, Any]], 
        source: str,
        weight: float = 1.0,
    ) -> None:
        """Add results from a search source.
        
        Args:
            results: List of result dictionaries
            source: Name of the source (e.g., "vector", "bm25")
            weight: Weight for this source's scores
        """
        for result in results:
            chunk_id = result["chunk_id"]
            score = result.get("score", 0) * weight
            
            if chunk_id not in self.results:
                self.results[chunk_id] = result.copy()
                self.results[chunk_id]["aggregated_score"] = score
                self.sources[chunk_id] = [source]
            else:
                # Combine scores
                self.results[chunk_id]["aggregated_score"] += score
                self.sources[chunk_id].append(source)
                # Keep the higher individual score
                if score > self.results[chunk_id].get("score", 0):
                    self.results[chunk_id]["score"] = result.get("score", 0)
    
    def get_aggregated_results(
        self, 
        top_k: int = 10,
        min_sources: int = 1,
    ) -> List[Dict[str, Any]]:
        """Get aggregated and ranked results.
        
        Args:
            top_k: Number of top results to return
            min_sources: Minimum number of sources required
            
        Returns:
            Aggregated and ranked results
        """
        # Filter by min_sources
        filtered = {
            k: v for k, v in self.results.items()
            if len(self.sources.get(k, [])) >= min_sources
        }
        
        # Sort by aggregated score
        sorted_results = sorted(
            filtered.items(),
            key=lambda x: x[1]["aggregated_score"],
            reverse=True,
        )
        
        # Build final results
        final = []
        for chunk_id, result in sorted_results[:top_k]:
            result_copy = result.copy()
            result_copy["sources"] = self.sources[chunk_id]
            result_copy["source_count"] = len(self.sources[chunk_id])
            final.append(result_copy)
        
        return final
    
    def reset(self) -> None:
        """Clear all aggregated results."""
        self.results.clear()
        self.sources.clear()
