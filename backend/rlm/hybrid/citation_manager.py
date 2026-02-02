"""Citation management for hybrid RAG.

This module provides citation tracking and management for retrieved chunks,
enabling source verification and citation-enhanced answer generation.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog

logger = structlog.get_logger()


@dataclass
class Citation:
    """Represents a citation to a source chunk."""
    
    citation_id: str
    chunk_id: str
    document_id: str
    content: str
    score: float
    citation_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def format_short(self) -> str:
        """Format as short citation reference."""
        return f"[{self.citation_number}]"
    
    def format_full(self) -> str:
        """Format as full citation with details."""
        preview = self.content[:100] + "..." if len(self.content) > 100 else self.content
        return (
            f"[{self.citation_number}] Document: {self.document_id}, "
            f"Score: {self.score:.3f}\n   {preview}"
        )


class CitationManager:
    """Manages citations for retrieved chunks.
    
    Tracks which chunks are used as sources and provides citation formatting
    for answer generation.
    
    Example:
        ```python
        manager = CitationManager()
        
        # Add chunks as citations
        for chunk in retrieved_chunks:
            manager.add_chunk_citation(
                chunk_id=chunk["chunk_id"],
                document_id=chunk["document_id"],
                content=chunk["content"],
                score=chunk["score"],
            )
        
        # Get citation summary
        citations = manager.format_citation_summary()
        ```
    """
    
    def __init__(self) -> None:
        """Initialize citation manager."""
        self.citations: Dict[str, Citation] = {}  # chunk_id -> Citation
        self.citation_counter = 0
        
        logger.info("citation_manager_initialized")
    
    def add_chunk_citation(
        self,
        chunk_id: str,
        document_id: str,
        content: str,
        score: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a chunk as a citation.
        
        Args:
            chunk_id: Unique chunk identifier
            document_id: Document identifier
            content: Chunk content
            score: Relevance score
            metadata: Optional additional metadata
            
        Returns:
            Citation ID
        """
        # Check if already added
        if chunk_id in self.citations:
            return self.citations[chunk_id].citation_id
        
        # Increment counter
        self.citation_counter += 1
        citation_number = self.citation_counter
        
        # Create citation
        citation = Citation(
            citation_id=f"cite_{citation_number}",
            chunk_id=chunk_id,
            document_id=document_id,
            content=content,
            score=score,
            citation_number=citation_number,
            metadata=metadata or {},
        )
        
        self.citations[chunk_id] = citation
        
        logger.debug(
            "citation_added",
            chunk_id=chunk_id,
            citation_number=citation_number,
            document_id=document_id,
        )
        
        return citation.citation_id
    
    def get_citation(self, chunk_id: str) -> Optional[Citation]:
        """Get citation for a chunk.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Citation if found, None otherwise
        """
        return self.citations.get(chunk_id)
    
    def get_citation_number(self, chunk_id: str) -> Optional[int]:
        """Get citation number for a chunk.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Citation number if found, None otherwise
        """
        citation = self.citations.get(chunk_id)
        return citation.citation_number if citation else None
    
    def format_citation_summary(
        self,
        format_type: str = "numbered",
        include_scores: bool = True,
        max_preview_length: int = 100,
    ) -> str:
        """Format citation summary for appending to answer.
        
        Args:
            format_type: Format type ("numbered", "document_grouped")
            include_scores: Whether to include relevance scores
            max_preview_length: Maximum length of content preview
            
        Returns:
            Formatted citation summary
        """
        if not self.citations:
            return ""
        
        # Sort by citation number
        sorted_citations = sorted(
            self.citations.values(),
            key=lambda c: c.citation_number,
        )
        
        if format_type == "numbered":
            return self._format_numbered(
                sorted_citations,
                include_scores,
                max_preview_length,
            )
        elif format_type == "document_grouped":
            return self._format_document_grouped(
                sorted_citations,
                include_scores,
                max_preview_length,
            )
        else:
            return self._format_numbered(
                sorted_citations,
                include_scores,
                max_preview_length,
            )
    
    def _format_numbered(
        self,
        citations: List[Citation],
        include_scores: bool,
        max_preview_length: int,
    ) -> str:
        """Format as numbered list."""
        lines = ["Sources:"]
        
        for citation in citations:
            preview = citation.content[:max_preview_length]
            if len(citation.content) > max_preview_length:
                preview += "..."
            
            if include_scores:
                lines.append(
                    f"[{citation.citation_number}] Doc: {citation.document_id} "
                    f"(Score: {citation.score:.3f}): {preview}"
                )
            else:
                lines.append(
                    f"[{citation.citation_number}] Doc: {citation.document_id}: {preview}"
                )
        
        return "\n".join(lines)
    
    def _format_document_grouped(
        self,
        citations: List[Citation],
        include_scores: bool,
        max_preview_length: int,
    ) -> str:
        """Format grouped by document."""
        # Group by document
        by_document: Dict[str, List[Citation]] = {}
        for citation in citations:
            doc_id = citation.document_id
            if doc_id not in by_document:
                by_document[doc_id] = []
            by_document[doc_id].append(citation)
        
        lines = ["Sources by Document:"]
        
        for doc_id, doc_citations in sorted(by_document.items()):
            lines.append(f"\nDocument: {doc_id}")
            
            for citation in doc_citations:
                preview = citation.content[:max_preview_length]
                if len(citation.content) > max_preview_length:
                    preview += "..."
                
                if include_scores:
                    lines.append(
                        f"  [{citation.citation_number}] Score: {citation.score:.3f}: {preview}"
                    )
                else:
                    lines.append(f"  [{citation.citation_number}]: {preview}")
        
        return "\n".join(lines)
    
    def extract_cited_chunks(self, text: str) -> Set[int]:
        """Extract citation numbers from text.
        
        Args:
            text: Text containing citations like [1], [2], etc.
            
        Returns:
            Set of cited citation numbers
        """
        # Find all [number] patterns
        pattern = r'\[(\d+)\]'
        matches = re.findall(pattern, text)
        
        return set(int(m) for m in matches)
    
    def verify_citations_in_answer(self, answer: str) -> Tuple[bool, List[int]]:
        """Verify that all citations in answer are valid.
        
        Args:
            answer: Generated answer with citations
            
        Returns:
            Tuple of (all_valid, list_of_invalid_citations)
        """
        cited = self.extract_cited_chunks(answer)
        valid_numbers = {c.citation_number for c in self.citations.values()}
        
        invalid = cited - valid_numbers
        
        if invalid:
            logger.warning(
                "invalid_citations_found",
                invalid_citations=list(invalid),
                valid_citations=list(valid_numbers),
            )
            return False, list(invalid)
        
        return True, []
    
    def get_citation_stats(self) -> Dict[str, Any]:
        """Get statistics about citations.
        
        Returns:
            Dictionary with citation statistics
        """
        if not self.citations:
            return {
                "total_citations": 0,
                "unique_documents": 0,
                "avg_score": 0.0,
            }
        
        scores = [c.score for c in self.citations.values()]
        documents = {c.document_id for c in self.citations.values()}
        
        return {
            "total_citations": len(self.citations),
            "unique_documents": len(documents),
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
        }
    
    def clear_citations(self) -> None:
        """Clear all citations."""
        self.citations.clear()
        self.citation_counter = 0
        
        logger.debug("citations_cleared")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert citations to dictionary format.
        
        Returns:
            Dictionary representation of citations
        """
        return {
            "citations": [
                {
                    "citation_id": c.citation_id,
                    "citation_number": c.citation_number,
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "score": c.score,
                    "content_preview": c.content[:200],
                    "metadata": c.metadata,
                }
                for c in sorted(
                    self.citations.values(),
                    key=lambda x: x.citation_number,
                )
            ],
            "stats": self.get_citation_stats(),
        }


class InlineCitationManager(CitationManager):
    """Extended citation manager with inline citation support.
    
    Provides functionality to automatically insert citation markers
    into generated text based on content similarity.
    
    Example:
        ```python
        manager = InlineCitationManager()
        
        # Add citations
        for chunk in chunks:
            manager.add_chunk_citation(...)
        
        # Auto-insert citations into answer
        cited_answer = manager.insert_citations(answer, min_similarity=0.7)
        ```
    """
    
    def insert_citations(
        self,
        text: str,
        min_similarity: float = 0.6,
        citation_format: str = "[{number}]",
    ) -> str:
        """Insert citation markers into text based on content overlap.
        
        Args:
            text: Text to add citations to
            min_similarity: Minimum similarity to add citation
            citation_format: Format string for citation (must contain {number})
            
        Returns:
            Text with inline citations
        """
        sentences = self._split_into_sentences(text)
        cited_sentences = []
        
        for sentence in sentences:
            # Find best matching citation
            best_citation = self._find_best_citation(sentence, min_similarity)
            
            if best_citation:
                citation_marker = citation_format.format(number=best_citation.citation_number)
                cited_sentences.append(f"{sentence} {citation_marker}")
            else:
                cited_sentences.append(sentence)
        
        return " ".join(cited_sentences)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _find_best_citation(
        self,
        sentence: str,
        min_similarity: float,
    ) -> Optional[Citation]:
        """Find best matching citation for a sentence."""
        best_citation = None
        best_score = min_similarity
        
        sentence_words = set(sentence.lower().split())
        
        for citation in self.citations.values():
            # Simple word overlap similarity
            citation_words = set(citation.content.lower().split())
            
            if not sentence_words or not citation_words:
                continue
            
            overlap = len(sentence_words & citation_words)
            similarity = overlap / len(sentence_words)
            
            if similarity > best_score:
                best_score = similarity
                best_citation = citation
        
        return best_citation
