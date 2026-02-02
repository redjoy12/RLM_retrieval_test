"""Document Ingestion Pipeline for RLM Document Retrieval System.

This module provides document parsing, text extraction, metadata extraction,
and storage capabilities for the RLM system. Supports 10+ document formats
and handles 10M+ token documents via efficient chunking.
"""

from rlm.documents.models import (
    BatchUploadResult,
    ChunkMetadata,
    ContentChunk,
    Document,
    DocumentContent,
    DocumentFormat,
    DocumentMetadata,
    IngestionOptions,
    ParseOptions,
    ProcessingProgress,
    ProcessingStatus,
)
from rlm.documents.ingestion import IngestionPipeline
from rlm.documents.format_detector import FormatDetector
from rlm.documents.storage import DocumentStorage
from rlm.documents.progress import ProgressTracker
from rlm.documents.cleaning import TextCleaner, clean_text
from rlm.documents.chunking import ChunkingEngine, SemanticChunker
from rlm.documents.metadata import MetadataExtractor, LLMMetadataExtractor

__all__ = [
    # Models
    "ChunkMetadata",
    "ContentChunk",
    "Document",
    "DocumentContent",
    "DocumentMetadata",
    "ProcessingStatus",
    "DocumentFormat",
    "ParseOptions",
    "IngestionOptions",
    "ProcessingProgress",
    "BatchUploadResult",
    # Main classes
    "IngestionPipeline",
    "FormatDetector",
    "DocumentStorage",
    "ProgressTracker",
    "TextCleaner",
    "clean_text",
    "ChunkingEngine",
    "SemanticChunker",
    "MetadataExtractor",
    "LLMMetadataExtractor",
]
