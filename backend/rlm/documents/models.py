"""Document data models for the ingestion pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class ProcessingStatus(str, Enum):
    """Document processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentFormat(str, Enum):
    """Supported document formats."""

    # Text formats
    PLAIN_TEXT = "txt"
    MARKDOWN = "md"

    # Document formats
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    PPTX = "pptx"

    # Web formats
    HTML = "html"
    XML = "xml"

    # Data formats
    CSV = "csv"
    JSON = "json"
    YAML = "yaml"

    # Archive formats
    ZIP = "zip"
    TAR = "tar"
    GZIP = "gz"

    # Code formats (various)
    PYTHON = "py"
    JAVASCRIPT = "js"
    TYPESCRIPT = "ts"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    GO = "go"
    RUST = "rs"
    RUBY = "rb"
    PHP = "php"

    # Notebook
    JUPYTER = "ipynb"

    # Unknown
    UNKNOWN = "unknown"

    @classmethod
    def from_extension(cls, ext: str) -> DocumentFormat:
        """Get format from file extension."""
        ext = ext.lower().lstrip(".")
        mapping = {
            "txt": cls.PLAIN_TEXT,
            "md": cls.MARKDOWN,
            "markdown": cls.MARKDOWN,
            "pdf": cls.PDF,
            "docx": cls.DOCX,
            "xlsx": cls.XLSX,
            "pptx": cls.PPTX,
            "html": cls.HTML,
            "htm": cls.HTML,
            "xml": cls.XML,
            "csv": cls.CSV,
            "json": cls.JSON,
            "yaml": cls.YAML,
            "yml": cls.YAML,
            "zip": cls.ZIP,
            "tar": cls.TAR,
            "gz": cls.GZIP,
            "py": cls.PYTHON,
            "js": cls.JAVASCRIPT,
            "ts": cls.TYPESCRIPT,
            "java": cls.JAVA,
            "cpp": cls.CPP,
            "c": cls.C,
            "go": cls.GO,
            "rs": cls.RUST,
            "rb": cls.RUBY,
            "php": cls.PHP,
            "ipynb": cls.JUPYTER,
        }
        return mapping.get(ext, cls.UNKNOWN)


class ChunkMetadata(BaseModel):
    """Metadata for a content chunk."""

    start_page: Optional[int] = None
    end_page: Optional[int] = None
    section_title: Optional[str] = None
    is_header: bool = False
    is_footer: bool = False
    is_table: bool = False
    is_code: bool = False
    language: Optional[str] = None
    custom: Dict[str, Any] = Field(default_factory=dict)


class ContentChunk(BaseModel):
    """A chunk of document content."""

    index: int = Field(..., description="Chunk index in the document")
    content: str = Field(..., description="Chunk text content")
    token_count: int = Field(default=0, description="Approximate token count")
    start_char: int = Field(..., description="Start character position")
    end_char: int = Field(..., description="End character position")
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)


class DocumentStructure(BaseModel):
    """Document structure information."""

    title: Optional[str] = None
    headings: List[str] = Field(default_factory=list)
    sections: List[Dict[str, Any]] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    figures: List[Dict[str, Any]] = Field(default_factory=list)


class DocumentMetadata(BaseModel):
    """Document metadata extracted during processing."""

    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    char_count: int = 0
    language: Optional[str] = None
    extraction_metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentContent(BaseModel):
    """Document content with chunks."""

    raw_text: str = Field(default="", description="Raw extracted text")
    cleaned_text: str = Field(default="", description="Cleaned text")
    chunks: List[ContentChunk] = Field(default_factory=list)
    structure: DocumentStructure = Field(default_factory=DocumentStructure)

    @property
    def total_tokens(self) -> int:
        """Calculate total token count across all chunks."""
        return sum(chunk.token_count for chunk in self.chunks)


class Document(BaseModel):
    """Complete document model."""

    id: UUID = Field(default_factory=uuid4)
    filename: str = Field(..., description="Original filename")
    file_path: Path = Field(..., description="Path to stored file")
    file_size: int = Field(..., description="File size in bytes")
    mime_type: str = Field(default="application/octet-stream")
    format_type: DocumentFormat = Field(..., description="Detected format")
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    content: DocumentContent = Field(default_factory=DocumentContent)
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("file_path", mode="before")
    @classmethod
    def validate_path(cls, v: Any) -> Path:
        """Ensure file_path is a Path object."""
        if isinstance(v, str):
            return Path(v)
        return v

    class Config:
        """Pydantic config."""

        json_encoders = {Path: str, datetime: lambda v: v.isoformat()}


class ParseOptions(BaseModel):
    """Options for document parsing."""

    extract_tables: bool = Field(default=True, description="Extract tables from documents")
    extract_images: bool = Field(default=False, description="Extract image descriptions")
    extract_metadata: bool = Field(default=True, description="Extract document metadata")
    preserve_formatting: bool = Field(default=True, description="Preserve formatting marks")
    language: Optional[str] = Field(default=None, description="Override language detection")
    password: Optional[str] = Field(default=None, description="Password for encrypted documents")
    timeout: int = Field(default=300, description="Parsing timeout in seconds")
    chunk_size: int = Field(default=100000, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=1000, description="Overlap between chunks")


class IngestionOptions(BaseModel):
    """Options for document ingestion."""

    parse_options: ParseOptions = Field(default_factory=ParseOptions)
    enable_cleaning: bool = Field(default=True, description="Enable text cleaning")
    enable_chunking: bool = Field(default=True, description="Enable content chunking")
    enable_metadata_extraction: bool = Field(default=True, description="Extract metadata")
    use_llm_metadata: bool = Field(default=False, description="Use LLM for metadata extraction")
    enable_embeddings: bool = Field(default=True, description="Generate embeddings for chunks")
    storage_backend: str = Field(default="local", description="Storage backend type")
    tags: Set[str] = Field(default_factory=set, description="User-defined tags")


class ProcessingProgress(BaseModel):
    """Progress information for document processing."""

    document_id: UUID
    status: ProcessingStatus
    stage: str = Field(default="", description="Current processing stage")
    progress_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    message: str = Field(default="", description="Status message")
    error: Optional[str] = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    stages_completed: List[str] = Field(default_factory=list)


class BatchUploadResult(BaseModel):
    """Result of batch upload operation."""

    successful: List[Document] = Field(default_factory=list)
    failed: List[Dict[str, Any]] = Field(default_factory=list)
    total_count: int = 0
    success_count: int = 0
    failed_count: int = 0
