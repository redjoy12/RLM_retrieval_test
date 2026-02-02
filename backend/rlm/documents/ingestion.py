"""Main document ingestion pipeline."""

import asyncio
from pathlib import Path
from typing import List, Optional
from uuid import UUID

from rlm.documents.cleaning import TextCleaner
from rlm.documents.chunking import ChunkingEngine
from rlm.documents.format_detector import FormatDetector
from rlm.documents.metadata import MetadataExtractor
from rlm.documents.models import (
    BatchUploadResult,
    Document,
    DocumentContent,
    DocumentFormat,
    IngestionOptions,
    ParseOptions,
    ProcessingStatus,
)
from rlm.documents.parsers import get_parser
from rlm.documents.parsers.text_parser import ParseError
from rlm.documents.progress import ProgressTracker
from rlm.documents.storage import DocumentStorage
from rlm.routing.rag_engine import RAGEngine


class IngestionPipeline:
    """Main document ingestion pipeline."""

    def __init__(
        self,
        storage: DocumentStorage,
        progress_tracker: Optional[ProgressTracker] = None,
    ) -> None:
        """
        Initialize ingestion pipeline.

        Args:
            storage: Document storage manager
            progress_tracker: Optional progress tracker
        """
        self.storage = storage
        self.progress = progress_tracker or ProgressTracker()
        self.format_detector = FormatDetector()
        self.metadata_extractor = MetadataExtractor()
        self.text_cleaner = TextCleaner()
        self.chunking_engine = ChunkingEngine()
        self.rag_engine: Optional[RAGEngine] = None  # Lazy initialization

    async def ingest(
        self,
        file_path: Path,
        filename: Optional[str] = None,
        options: Optional[IngestionOptions] = None,
    ) -> Document:
        """
        Ingest a single document.

        Args:
            file_path: Path to the file to ingest
            filename: Original filename (if different from file_path)
            options: Ingestion options

        Returns:
            Processed document
        """
        options = options or IngestionOptions()
        file_path = Path(file_path)
        filename = filename or file_path.name

        # Create document record
        document = Document(
            filename=filename,
            file_path=file_path,
            file_size=file_path.stat().st_size,
            format_type=DocumentFormat.UNKNOWN,
            status=ProcessingStatus.PENDING,
        )

        # Start progress tracking
        self.progress.start_tracking(document.id)

        try:
            # Stage 1: Validate and detect format
            await self._stage_validate(document, options)

            # Stage 2: Save file to storage
            await self._stage_save_file(document, file_path, options)

            # Stage 3: Parse document
            await self._stage_parse(document, options)

            # Stage 4: Clean content
            if options.enable_cleaning:
                await self._stage_clean(document, options)

            # Stage 5: Extract metadata
            if options.enable_metadata_extraction:
                await self._stage_extract_metadata(document, options)

            # Stage 6: Chunk content
            if options.enable_chunking:
                await self._stage_chunk(document, options)

            # Stage 7: Generate embeddings
            if options.enable_embeddings and options.enable_chunking:
                await self._stage_embed(document, options)

            # Stage 8: Save to storage
            await self._stage_save_document(document, options)

            # Mark as completed
            document.status = ProcessingStatus.COMPLETED
            self.progress.mark_completed(document.id)

        except Exception as e:
            document.status = ProcessingStatus.FAILED
            document.error_message = str(e)
            self.progress.mark_failed(document.id, str(e))

        # Save final state
        await self.storage.save_document(document)

        return document

    async def ingest_batch(
        self,
        file_paths: List[Path],
        options: Optional[IngestionOptions] = None,
    ) -> BatchUploadResult:
        """
        Ingest multiple documents in batch.

        Args:
            file_paths: List of file paths to ingest
            options: Ingestion options

        Returns:
            Batch upload result
        """
        options = options or IngestionOptions()
        result = BatchUploadResult()
        result.total_count = len(file_paths)

        # Process files concurrently (with limit)
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent uploads

        async def ingest_with_limit(file_path: Path) -> None:
            async with semaphore:
                try:
                    doc = await self.ingest(file_path, options=options)
                    if doc.status == ProcessingStatus.COMPLETED:
                        result.successful.append(doc)
                        result.success_count += 1
                    else:
                        result.failed.append(
                            {"file": str(file_path), "error": doc.error_message}
                        )
                        result.failed_count += 1
                except Exception as e:
                    result.failed.append({"file": str(file_path), "error": str(e)})
                    result.failed_count += 1

        # Process all files
        await asyncio.gather(*[ingest_with_limit(fp) for fp in file_paths])

        return result

    async def _stage_validate(
        self, document: Document, options: IngestionOptions
    ) -> None:
        """Validate and detect format."""
        self.progress.update_progress(
            document.id,
            "validation",
            5.0,
            "Validating file and detecting format",
        )

        # Detect format
        format_type, mime_type = self.format_detector.detect(document.file_path)

        if format_type == DocumentFormat.UNKNOWN:
            raise ParseError(f"Unsupported file format: {document.file_path}")

        document.format_type = format_type
        document.mime_type = mime_type

        # Check if parser available
        parser = get_parser(format_type)
        if not parser:
            raise ParseError(f"No parser available for format: {format_type}")

    async def _stage_save_file(
        self, document: Document, file_path: Path, options: IngestionOptions
    ) -> None:
        """Save file to storage."""
        self.progress.update_progress(
            document.id,
            "storage",
            10.0,
            "Saving file to storage",
        )

        stored_path = await self.storage.save_file(
            file_path, document.id, document.filename
        )
        document.file_path = stored_path

    async def _stage_parse(
        self, document: Document, options: IngestionOptions
    ) -> None:
        """Parse document content."""
        self.progress.update_progress(
            document.id,
            "parsing",
            30.0,
            f"Parsing document with {document.format_type.value} parser",
        )

        parser = get_parser(document.format_type)
        if not parser:
            raise ParseError(f"Parser not found for format: {document.format_type}")

        parse_options = options.parse_options or ParseOptions()
        document.content = await parser.parse(document.file_path, parse_options)

    async def _stage_clean(
        self, document: Document, options: IngestionOptions
    ) -> None:
        """Clean document content."""
        self.progress.update_progress(
            document.id,
            "cleaning",
            50.0,
            "Cleaning and normalizing text",
        )

        cleaner = TextCleaner()
        document.content.cleaned_text = cleaner.clean(
            document.content.raw_text
        )

    async def _stage_extract_metadata(
        self, document: Document, options: IngestionOptions
    ) -> None:
        """Extract document metadata."""
        self.progress.update_progress(
            document.id,
            "metadata",
            70.0,
            "Extracting metadata",
        )

        content = document.content.cleaned_text or document.content.raw_text
        document.metadata = await self.metadata_extractor.extract(
            document.file_path,
            content,
            document.format_type,
        )

        # Update char/word counts
        document.metadata.char_count = len(content)
        document.metadata.word_count = len(content.split())

    async def _stage_chunk(
        self, document: Document, options: IngestionOptions
    ) -> None:
        """Chunk document content."""
        self.progress.update_progress(
            document.id,
            "chunking",
            85.0,
            "Creating content chunks",
        )

        content = document.content.cleaned_text or document.content.raw_text
        parse_options = options.parse_options or ParseOptions()

        chunks = self.chunking_engine.chunk(content, parse_options)
        document.content.chunks = chunks

    async def _stage_save_document(
        self, document: Document, options: IngestionOptions
    ) -> None:
        """Save document to storage."""
        self.progress.update_progress(
            document.id,
            "saving",
            95.0,
            "Saving document metadata",
        )

        await self.storage.save_document(document)

    async def _stage_embed(
        self, document: Document, options: IngestionOptions
    ) -> None:
        """Generate embeddings for document chunks."""
        self.progress.update_progress(
            document.id,
            "embedding",
            90.0,
            "Generating embeddings for chunks",
        )

        if not document.content.chunks:
            return

        # Initialize RAG engine if needed
        if self.rag_engine is None:
            self.rag_engine = RAGEngine()

        # Prepare chunks for embedding
        chunk_data = [
            {
                "content": chunk.content,
                "index": chunk.index,
            }
            for chunk in document.content.chunks
        ]

        # Add chunks to vector store
        await self.rag_engine.add_chunks(
            document_id=str(document.id),
            chunks=chunk_data,
        )

        # Store embedding metadata on document
        document.metadata.embedding_model = self.rag_engine.embedding_model
        document.metadata.embedding_count = len(chunk_data)
