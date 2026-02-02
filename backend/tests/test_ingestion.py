"""Document ingestion pipeline tests."""

import pytest
from pathlib import Path
from uuid import UUID

from rlm.documents.models import Document, DocumentFormat, ProcessingStatus
from rlm.documents.format_detector import FormatDetector
from rlm.documents.cleaning import TextCleaner
from rlm.documents.chunking import ChunkingEngine
from rlm.documents.storage import DocumentStorage
from rlm.documents.ingestion import IngestionPipeline


class TestFormatDetector:
    """Test format detection."""

    def test_detect_text_file(self, tmp_path):
        """Test detecting text file format."""
        detector = FormatDetector()
        
        # Create a text file
        text_file = tmp_path / "test.txt"
        text_file.write_text("This is a test file.")
        
        format_type, mime = detector.detect(text_file)
        assert format_type == DocumentFormat.PLAIN_TEXT

    def test_detect_markdown_file(self, tmp_path):
        """Test detecting markdown file format."""
        detector = FormatDetector()
        
        md_file = tmp_path / "test.md"
        md_file.write_text("# Header\n\nContent")
        
        format_type, mime = detector.detect(md_file)
        assert format_type == DocumentFormat.MARKDOWN


class TestTextCleaner:
    """Test text cleaning."""

    def test_clean_whitespace(self):
        """Test whitespace normalization."""
        cleaner = TextCleaner()
        
        text = "Line 1\n\n\n\nLine 2"
        cleaned = cleaner.clean(text)
        
        assert "\n\n\n" not in cleaned

    def test_fix_encoding(self):
        """Test encoding fixes."""
        cleaner = TextCleaner(fix_encoding=True)
        
        # Text with potential encoding issues
        text = "caf\u00e9"  # Properly encoded é
        cleaned = cleaner.clean(text)
        
        assert "caf" in cleaned


class TestChunkingEngine:
    """Test content chunking."""

    def test_chunk_small_text(self):
        """Test chunking small text."""
        engine = ChunkingEngine(chunk_size=100)
        
        text = "This is a test. " * 5
        chunks = engine.chunk(text)
        
        assert len(chunks) == 1
        assert chunks[0].index == 0

    def test_chunk_large_text(self):
        """Test chunking large text."""
        engine = ChunkingEngine(chunk_size=50)
        
        # Create text larger than chunk size
        text = "Word " * 50
        chunks = engine.chunk(text)
        
        assert len(chunks) > 1
        
        # Check chunk boundaries
        for i, chunk in enumerate(chunks):
            assert chunk.index == i
            assert len(chunk.content) <= 50 + 20  # Allow some flexibility


class TestDocumentStorage:
    """Test document storage."""

    @pytest.fixture
    async def storage(self, tmp_path):
        """Create temporary storage."""
        return DocumentStorage(tmp_path)

    @pytest.mark.asyncio
    async def test_save_and_get_document(self, tmp_path):
        """Test saving and retrieving document."""
        storage = DocumentStorage(tmp_path)
        
        # Create a document
        doc = Document(
            filename="test.txt",
            file_path=tmp_path / "test.txt",
            file_size=100,
            format_type=DocumentFormat.PLAIN_TEXT,
            status=ProcessingStatus.COMPLETED,
        )
        
        # Save document
        await storage.save_document(doc)
        
        # Retrieve document
        retrieved = await storage.get_document(doc.id)
        
        assert retrieved is not None
        assert retrieved.filename == doc.filename
        assert retrieved.id == doc.id

    @pytest.mark.asyncio
    async def test_list_documents(self, tmp_path):
        """Test listing documents."""
        storage = DocumentStorage(tmp_path)
        
        # Create and save multiple documents
        for i in range(3):
            doc = Document(
                filename=f"test{i}.txt",
                file_path=tmp_path / f"test{i}.txt",
                file_size=100,
                format_type=DocumentFormat.PLAIN_TEXT,
                status=ProcessingStatus.COMPLETED,
            )
            await storage.save_document(doc)
        
        # List documents
        documents = await storage.list_documents(limit=10)
        
        assert len(documents) == 3

    @pytest.mark.asyncio
    async def test_delete_document(self, tmp_path):
        """Test deleting document."""
        storage = DocumentStorage(tmp_path)
        
        # Create and save document
        doc = Document(
            filename="test.txt",
            file_path=tmp_path / "test.txt",
            file_size=100,
            format_type=DocumentFormat.PLAIN_TEXT,
        )
        await storage.save_document(doc)
        
        # Delete document
        deleted = await storage.delete_document(doc.id)
        
        assert deleted is True
        
        # Verify deletion
        retrieved = await storage.get_document(doc.id)
        assert retrieved is None


class TestIngestionPipeline:
    """Test ingestion pipeline."""

    @pytest.mark.asyncio
    async def test_ingest_text_file(self, tmp_path):
        """Test ingesting a text file."""
        storage = DocumentStorage(tmp_path)
        pipeline = IngestionPipeline(storage)
        
        # Create a text file
        text_file = tmp_path / "document.txt"
        text_file.write_text("This is a test document.\n\nIt has multiple paragraphs.")
        
        # Ingest
        document = await pipeline.ingest(text_file)
        
        assert document.status == ProcessingStatus.COMPLETED
        assert document.format_type == DocumentFormat.PLAIN_TEXT
        assert len(document.content.chunks) > 0

    @pytest.mark.asyncio
    async def test_ingest_markdown_file(self, tmp_path):
        """Test ingesting a markdown file."""
        storage = DocumentStorage(tmp_path)
        pipeline = IngestionPipeline(storage)
        
        # Create a markdown file
        md_file = tmp_path / "document.md"
        md_file.write_text("# Title\n\n## Section 1\n\nContent here.")
        
        # Ingest
        document = await pipeline.ingest(md_file)
        
        assert document.status == ProcessingStatus.COMPLETED
        assert document.format_type == DocumentFormat.MARKDOWN
        assert document.content.structure.title == "Title"
