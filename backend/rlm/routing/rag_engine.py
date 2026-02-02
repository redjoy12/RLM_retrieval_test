"""RAG engine using Qdrant for vector retrieval.

Enhanced with hybrid search capabilities (semantic + keyword) and reranking support.
"""

import time
import uuid
from typing import Any, Dict, List, Optional

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

from rlm.config import get_settings
from rlm.routing.models import RAGChunk, RAGResult

# Import hybrid components (Component 8)
try:
    from rlm.hybrid.search_engines import HybridSearcher
    from rlm.hybrid.reranker import CrossEncoderReranker, RerankerPipeline
    from rlm.hybrid.chunk_selector import AdaptiveChunkSelector
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False

logger = structlog.get_logger()


class RAGEngine:
    """Retrieval-Augmented Generation engine using Qdrant.
    
    Provides vector-based semantic search over document chunks
    with filtering capabilities.
    
    Example:
        ```python
        engine = RAGEngine()
        result = await engine.retrieve(
            query="What is machine learning?",
            document_ids=["doc-123"],
            top_k=5
        )
        for chunk in result.chunks:
            print(f"Score: {chunk.score}, Content: {chunk.content[:100]}")
        ```
    """
    
    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        """Initialize RAG engine.
        
        Args:
            client: Qdrant client (created if None)
            collection_name: Name of Qdrant collection
            embedding_model: Model for generating embeddings
        """
        self.settings = get_settings()
        
        # Initialize Qdrant client
        if client:
            self.client = client
        else:
            # Get connection settings
            host = getattr(self.settings, 'qdrant_host', 'localhost')
            port = getattr(self.settings, 'qdrant_port', 6333)
            self.client = QdrantClient(host=host, port=port)
        
        self.collection_name = collection_name or getattr(
            self.settings, 'qdrant_collection', 'rlm_chunks'
        )
        self.embedding_model = embedding_model or getattr(
            self.settings, 'embedding_model', 'text-embedding-3-small'
        )
        
        # Vector size for text-embedding-3-small
        self.vector_size = 1536
        
        # Ensure collection exists
        self._ensure_collection()
        
        logger.info(
            "rag_engine_initialized",
            collection=self.collection_name,
            model=self.embedding_model,
        )
    
    def _ensure_collection(self) -> None:
        """Ensure the Qdrant collection exists."""
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(
                    "qdrant_collection_created",
                    collection=self.collection_name,
                )
        except Exception as e:
            logger.error(
                "failed_to_ensure_collection",
                collection=self.collection_name,
                error=str(e),
            )
            raise
    
    async def retrieve(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        use_hybrid: bool = False,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> RAGResult:
        """Retrieve relevant chunks for a query.
        
        Args:
            query: The search query
            document_ids: Filter to specific documents (None = all)
            top_k: Number of chunks to retrieve
            score_threshold: Minimum similarity score (0-1)
            use_hybrid: Whether to use hybrid search (semantic + keyword)
            semantic_weight: Weight for semantic search (0-1)
            keyword_weight: Weight for keyword search (0-1)
            
        Returns:
            RAGResult with retrieved chunks
        """
        start_time = time.time()
        
        # Use hybrid search if enabled and available
        if use_hybrid and HYBRID_AVAILABLE:
            return await self._hybrid_retrieve(
                query=query,
                document_ids=document_ids,
                top_k=top_k,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
            )
        
        # Standard vector retrieval
        query_embedding = await self._generate_embedding(query)
        
        # Build filter
        query_filter = None
        if document_ids:
            # Filter by document IDs
            should_conditions = [
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=doc_id),
                )
                for doc_id in document_ids
            ]
            query_filter = Filter(should=should_conditions)
        
        # Search
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold,
            )
            
            # Convert to RAGChunks
            chunks = []
            for point in search_result:
                chunk = RAGChunk(
                    chunk_id=point.id,
                    document_id=point.payload.get("document_id", "unknown"),
                    content=point.payload.get("content", ""),
                    score=point.score,
                    metadata={
                        "chunk_index": point.payload.get("chunk_index"),
                        "char_count": point.payload.get("char_count"),
                    },
                )
                chunks.append(chunk)
            
            retrieval_time = (time.time() - start_time) * 1000
            
            logger.info(
                "rag_retrieval_complete",
                query=query[:50],
                chunks_found=len(chunks),
                retrieval_time_ms=retrieval_time,
                hybrid_mode=False,
            )
            
            return RAGResult(
                query=query,
                chunks=chunks,
                total_chunks_searched=len(search_result),
                retrieval_time_ms=retrieval_time,
            )
            
        except Exception as e:
            logger.error(
                "rag_retrieval_failed",
                query=query[:50],
                error=str(e),
            )
            raise
    
    async def _hybrid_retrieve(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> RAGResult:
        """Perform hybrid retrieval combining semantic and keyword search.
        
        Args:
            query: The search query
            document_ids: Filter to specific documents
            top_k: Number of chunks to retrieve
            semantic_weight: Weight for semantic search
            keyword_weight: Weight for keyword search
            
        Returns:
            RAGResult with hybrid-retrieved chunks
        """
        start_time = time.time()
        
        if not HYBRID_AVAILABLE:
            logger.warning("hybrid_search_not_available, falling_back_to_vector")
            # Fall back to standard retrieval
            return await self.retrieve(
                query=query,
                document_ids=document_ids,
                top_k=top_k,
                use_hybrid=False,
            )
        
        # Initialize hybrid searcher
        hybrid_searcher = HybridSearcher(
            qdrant_client=self.client,
            collection_name=self.collection_name,
            embedding_model=self.embedding_model,
        )
        
        try:
            # Perform hybrid search
            hybrid_results = await hybrid_searcher.search(
                query=query,
                document_ids=document_ids,
                top_k=top_k,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
                vector_top_k=top_k * 2,
                keyword_top_k=top_k * 2,
            )
            
            # Convert to RAGChunks
            chunks = []
            for result in hybrid_results:
                chunk = RAGChunk(
                    chunk_id=result["chunk_id"],
                    document_id=result["document_id"],
                    content=result["content"],
                    score=result.get("fused_score", result.get("score", 0)),
                    metadata={
                        "chunk_index": result.get("chunk_index"),
                        "search_type": result.get("search_type", "hybrid"),
                        "rrf_rank": result.get("rrf_rank"),
                    },
                )
                chunks.append(chunk)
            
            retrieval_time = (time.time() - start_time) * 1000
            
            logger.info(
                "hybrid_retrieval_complete",
                query=query[:50],
                chunks_found=len(chunks),
                retrieval_time_ms=retrieval_time,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
            )
            
            return RAGResult(
                query=query,
                chunks=chunks,
                total_chunks_searched=len(hybrid_results),
                retrieval_time_ms=retrieval_time,
            )
            
        except Exception as e:
            logger.error(
                "hybrid_retrieval_failed",
                query=query[:50],
                error=str(e),
            )
            # Fall back to standard retrieval
            logger.info("falling_back_to_vector_search")
            return await self.retrieve(
                query=query,
                document_ids=document_ids,
                top_k=top_k,
                use_hybrid=False,
            )
    
    async def add_chunks(
        self,
        document_id: str,
        chunks: List[Dict[str, any]],
    ) -> None:
        """Add document chunks to the vector store.
        
        Args:
            document_id: Document identifier
            chunks: List of chunk dicts with 'content' and optionally 'index'
        """
        points = []
        
        for i, chunk_data in enumerate(chunks):
            content = chunk_data.get("content", "")
            chunk_index = chunk_data.get("index", i)
            
            # Generate embedding
            embedding = await self._generate_embedding(content)
            
            # Create point
            point_id = str(uuid.uuid4())
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "document_id": document_id,
                    "content": content,
                    "chunk_index": chunk_index,
                    "char_count": len(content),
                },
            )
            points.append(point)
        
        # Batch upload
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            
            logger.info(
                "chunks_added_to_vector_store",
                document_id=document_id,
                chunk_count=len(points),
            )
            
        except Exception as e:
            logger.error(
                "failed_to_add_chunks",
                document_id=document_id,
                error=str(e),
            )
            raise
    
    async def retrieve_with_reranking(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5,
        rerank_top_k: int = 20,
        use_hybrid: bool = True,
        enable_reranking: bool = True,
    ) -> RAGResult:
        """Retrieve with advanced reranking pipeline.
        
        This method performs hybrid retrieval followed by cross-encoder reranking
        and adaptive chunk selection for optimal results.
        
        Args:
            query: The search query
            document_ids: Filter to specific documents
            top_k: Final number of chunks to return
            rerank_top_k: Number of chunks to rerank
            use_hybrid: Whether to use hybrid search
            enable_reranking: Whether to apply reranking
            
        Returns:
            RAGResult with reranked chunks
        """
        start_time = time.time()
        
        if not HYBRID_AVAILABLE or not enable_reranking:
            # Fall back to standard retrieval
            return await self.retrieve(
                query=query,
                document_ids=document_ids,
                top_k=top_k,
                use_hybrid=use_hybrid,
            )
        
        try:
            # Step 1: Initial retrieval (get more for reranking)
            initial_result = await self.retrieve(
                query=query,
                document_ids=document_ids,
                top_k=rerank_top_k,
                use_hybrid=use_hybrid,
            )
            
            if not initial_result.chunks:
                return initial_result
            
            # Step 2: Convert to format for reranking
            chunks_for_rerank = [
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "score": chunk.score,
                    "metadata": chunk.metadata,
                }
                for chunk in initial_result.chunks
            ]
            
            # Step 3: Apply reranking
            reranker_pipeline = RerankerPipeline(
                reranker=CrossEncoderReranker(),
                rerank_top_k=rerank_top_k,
                final_top_k=top_k,
            )
            
            reranked_chunks = await reranker_pipeline.rerank(query, chunks_for_rerank)
            
            # Step 4: Apply adaptive selection
            selector = AdaptiveChunkSelector(
                min_chunks=3,
                max_chunks=top_k,
                diversity_threshold=0.8,
            )
            
            selected_chunks = selector.select(
                chunks=reranked_chunks,
                query=query,
                max_chunks=top_k,
            )
            
            # Convert back to RAGChunks
            final_chunks = []
            for chunk_data in selected_chunks:
                chunk = RAGChunk(
                    chunk_id=chunk_data["chunk_id"],
                    document_id=chunk_data["document_id"],
                    content=chunk_data["content"],
                    score=chunk_data.get("cross_encoder_score", chunk_data.get("score", 0)),
                    metadata={
                        **chunk_data.get("metadata", {}),
                        "reranked": True,
                        "original_score": chunk_data.get("score"),
                    },
                )
                final_chunks.append(chunk)
            
            retrieval_time = (time.time() - start_time) * 1000
            
            logger.info(
                "reranked_retrieval_complete",
                query=query[:50],
                initial_chunks=len(initial_result.chunks),
                final_chunks=len(final_chunks),
                retrieval_time_ms=retrieval_time,
            )
            
            return RAGResult(
                query=query,
                chunks=final_chunks,
                total_chunks_searched=initial_result.total_chunks_searched,
                retrieval_time_ms=retrieval_time,
            )
            
        except Exception as e:
            logger.error(
                "reranked_retrieval_failed",
                query=query[:50],
                error=str(e),
            )
            # Fall back to standard retrieval
            return await self.retrieve(
                query=query,
                document_ids=document_ids,
                top_k=top_k,
                use_hybrid=use_hybrid,
            )
    
    async def retrieve_adaptive(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        max_chunks: int = 10,
        use_hybrid: bool = True,
        enable_reranking: bool = True,
    ) -> RAGResult:
        """Adaptive retrieval with automatic parameter selection.
        
        Automatically determines optimal chunk count based on query complexity.
        
        Args:
            query: The search query
            document_ids: Filter to specific documents
            max_chunks: Maximum chunks to return
            use_hybrid: Whether to use hybrid search
            enable_reranking: Whether to apply reranking
            
        Returns:
            RAGResult with adaptively selected chunks
        """
        if not HYBRID_AVAILABLE:
            return await self.retrieve(
                query=query,
                document_ids=document_ids,
                top_k=max_chunks,
                use_hybrid=False,
            )
        
        # Use selector to determine optimal chunk count
        selector = AdaptiveChunkSelector(
            min_chunks=3,
            max_chunks=max_chunks,
        )
        
        # First get a larger set for analysis
        temp_result = await self.retrieve(
            query=query,
            document_ids=document_ids,
            top_k=max_chunks * 2,
            use_hybrid=use_hybrid,
        )
        
        if not temp_result.chunks:
            return temp_result
        
        # Convert for selector
        chunks_data = [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "content": c.content,
                "score": c.score,
            }
            for c in temp_result.chunks
        ]
        
        # Get adaptive count
        target_count = selector._compute_target_count(
            len(chunks_data),
            selector._estimate_query_complexity(query),
            max_chunks,
        )
        
        # Retrieve with reranking using adaptive count
        return await self.retrieve_with_reranking(
            query=query,
            document_ids=document_ids,
            top_k=target_count,
            rerank_top_k=max_chunks * 2,
            use_hybrid=use_hybrid,
            enable_reranking=enable_reranking,
        )
    
    async def delete_document_chunks(self, document_id: str) -> int:
        """Delete all chunks for a document.
        
        Args:
            document_id: Document to delete
            
        Returns:
            Number of chunks deleted
        """
        try:
            # First, find all points for this document
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            )
            
            # Scroll to get all points
            points = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=1000,  # Batch size
            )[0]
            
            point_ids = [p.id for p in points]
            
            # Delete points
            if point_ids:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=point_ids,
                )
            
            logger.info(
                "document_chunks_deleted",
                document_id=document_id,
                deleted_count=len(point_ids),
            )
            
            return len(point_ids)
            
        except Exception as e:
            logger.error(
                "failed_to_delete_document_chunks",
                document_id=document_id,
                error=str(e),
            )
            raise
    
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
            return [0.0] * self.vector_size
    
    def get_collection_stats(self) -> Dict[str, any]:
        """Get statistics about the vector collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            info = self.client.get_collection(self.collection_name)
            
            # Count points
            count = self.client.count(self.collection_name)
            
            return {
                "collection_name": self.collection_name,
                "vector_size": info.config.params.vectors.size,
                "distance_metric": str(info.config.params.vectors.distance),
                "total_chunks": count.count,
            }
            
        except Exception as e:
            logger.error(
                "failed_to_get_collection_stats",
                error=str(e),
            )
            return {
                "collection_name": self.collection_name,
                "error": str(e),
            }


class RAGSearcher:
    """High-level RAG search interface."""
    
    def __init__(self, engine: Optional[RAGEngine] = None) -> None:
        """Initialize RAG searcher."""
        self.engine = engine or RAGEngine()
    
    async def search(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, any]]:
        """Simple search interface returning chunk data.
        
        Args:
            query: Search query
            document_ids: Optional document filter
            top_k: Number of results
            
        Returns:
            List of chunk dictionaries
        """
        result = await self.engine.retrieve(
            query=query,
            document_ids=document_ids,
            top_k=top_k,
        )
        
        return [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "score": chunk.score,
                "metadata": chunk.metadata,
            }
            for chunk in result.chunks
        ]
