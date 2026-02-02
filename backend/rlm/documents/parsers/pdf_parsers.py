"""PDF document parsers using PyMuPDF and Docling."""

import asyncio
from pathlib import Path
from typing import List, Optional, Set

import fitz  # PyMuPDF

from rlm.documents.models import (
    ChunkMetadata,
    ContentChunk,
    DocumentContent,
    DocumentFormat,
    DocumentMetadata,
    DocumentStructure,
    ParseOptions,
)
from rlm.documents.parsers.base import DocumentParser
from rlm.documents.parsers.text_parser import ParseError


class PyMuPDFParser(DocumentParser):
    """PDF parser using PyMuPDF (fitz)."""

    SUPPORTED_FORMATS = {DocumentFormat.PDF}

    @property
    def name(self) -> str:
        return "pymupdf"

    @property
    def priority(self) -> int:
        return 80  # Lower priority than docling for academic papers

    def supports(self, format_type: DocumentFormat) -> bool:
        return format_type in self.SUPPORTED_FORMATS

    async def parse(
        self, file_path: Path, options: ParseOptions
    ) -> DocumentContent:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._parse_sync, file_path, options
        )

    def _parse_sync(self, file_path: Path, options: ParseOptions) -> DocumentContent:
        """Synchronous PDF parsing."""
        try:
            doc = fitz.open(file_path)

            if options.password:
                doc.authenticate(options.password)

            full_text = []
            chunks = []
            structure = DocumentStructure()
            char_pos = 0

            # Extract metadata
            metadata = self._extract_metadata(doc)

            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]

                # Extract text
                text = page.get_text()
                full_text.append(text)

                # Extract structure (headings) from first page
                if page_num == 0:
                    structure = self._extract_structure(text)

                # Create chunk for this page
                page_chunk = ContentChunk(
                    index=page_num,
                    content=text,
                    start_char=char_pos,
                    end_char=char_pos + len(text),
                    metadata=ChunkMetadata(
                        start_page=page_num + 1,
                        end_page=page_num + 1,
                        is_table=False,
                    ),
                )
                chunks.append(page_chunk)
                char_pos += len(text)

                # Extract tables if requested
                if options.extract_tables:
                    tables = page.find_tables()
                    for table_idx, table in enumerate(tables):
                        table_text = self._table_to_text(table)
                        table_chunk = ContentChunk(
                            index=len(chunks),
                            content=f"[TABLE {table_idx + 1}]\n{table_text}",
                            start_char=char_pos,
                            end_char=char_pos + len(table_text) + 20,
                            metadata=ChunkMetadata(
                                start_page=page_num + 1,
                                end_page=page_num + 1,
                                is_table=True,
                            ),
                        )
                        chunks.append(table_chunk)

            doc.close()

            raw_text = "\n\n".join(full_text)

            # Apply chunking strategy if needed
            if len(chunks) == 0 or options.chunk_size > 0:
                chunks = self._rechunk_content(raw_text, options)

            return DocumentContent(
                raw_text=raw_text,
                cleaned_text=raw_text,
                chunks=chunks,
                structure=structure,
            )

        except Exception as e:
            raise ParseError(f"Failed to parse PDF with PyMuPDF: {e}") from e

    def _extract_metadata(self, doc: fitz.Document) -> DocumentMetadata:
        """Extract PDF metadata."""
        meta = doc.metadata
        return DocumentMetadata(
            title=meta.get("title"),
            author=meta.get("author"),
            subject=meta.get("subject"),
            page_count=len(doc),
        )

    def _extract_structure(self, text: str) -> DocumentStructure:
        """Extract document structure from text."""
        structure = DocumentStructure()
        lines = text.split("\n")

        # Heuristic: first non-empty line might be title
        for line in lines[:10]:
            line = line.strip()
            if line and len(line) < 200:
                structure.title = line
                break

        return structure

    def _table_to_text(self, table) -> str:
        """Convert table to text representation."""
        rows = []
        for row in table.extract():
            row_text = " | ".join(str(cell) if cell else "" for cell in row)
            rows.append(row_text)
        return "\n".join(rows)

    def _rechunk_content(
        self, text: str, options: ParseOptions
    ) -> List[ContentChunk]:
        """Rechunk content based on options."""
        chunks = []
        chunk_size = options.chunk_size
        overlap = options.chunk_overlap

        paragraphs = text.split("\n\n")
        current_chunk = ""
        char_pos = 0
        chunk_idx = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_with_newline = para + "\n\n"

            if len(current_chunk) + len(para_with_newline) > chunk_size:
                if current_chunk:
                    chunks.append(
                        ContentChunk(
                            index=chunk_idx,
                            content=current_chunk.strip(),
                            start_char=char_pos,
                            end_char=char_pos + len(current_chunk),
                        )
                    )
                    chunk_idx += 1
                    char_pos += len(current_chunk) - overlap

                if overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-overlap:]
                    current_chunk = overlap_text + para_with_newline
                else:
                    current_chunk = para_with_newline
            else:
                current_chunk += para_with_newline

        if current_chunk.strip():
            chunks.append(
                ContentChunk(
                    index=chunk_idx,
                    content=current_chunk.strip(),
                    start_char=char_pos,
                    end_char=char_pos + len(current_chunk),
                )
            )

        return chunks


class DoclingParser(DocumentParser):
    """PDF parser using Docling for academic/structured documents."""

    SUPPORTED_FORMATS = {DocumentFormat.PDF}

    def __init__(self) -> None:
        """Initialize docling parser."""
        self._docling_available = False
        try:
            from docling.document_converter import DocumentConverter

            self._converter = DocumentConverter()
            self._docling_available = True
        except ImportError:
            self._converter = None

    @property
    def name(self) -> str:
        return "docling"

    @property
    def priority(self) -> int:
        return 100  # Higher priority for academic papers

    def supports(self, format_type: DocumentFormat) -> bool:
        if not self._docling_available:
            return False
        return format_type in self.SUPPORTED_FORMATS

    async def parse(
        self, file_path: Path, options: ParseOptions
    ) -> DocumentContent:
        if not self._docling_available:
            raise ParseError("Docling is not installed")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._parse_sync, file_path, options
        )

    def _parse_sync(self, file_path: Path, options: ParseOptions) -> DocumentContent:
        """Synchronous parsing with docling."""
        try:
            result = self._converter.convert(str(file_path))
            doc = result.document

            # Export to markdown
            markdown_text = doc.export_to_markdown()

            # Extract structure
            structure = self._extract_structure(doc)

            # Create chunks
            chunks = self._create_chunks(doc, markdown_text, options)

            return DocumentContent(
                raw_text=markdown_text,
                cleaned_text=markdown_text,
                chunks=chunks,
                structure=structure,
            )

        except Exception as e:
            raise ParseError(f"Failed to parse PDF with Docling: {e}") from e

    def _extract_structure(self, doc) -> DocumentStructure:
        """Extract document structure from docling document."""
        structure = DocumentStructure()

        # Extract title
        if hasattr(doc, "title") and doc.title:
            structure.title = doc.title

        # Extract headings and sections
        for item in doc.texts:
            if hasattr(item, "label"):
                if item.label == "section_header":
                    structure.headings.append(str(item))
                elif item.label == "title":
                    if not structure.title:
                        structure.title = str(item)

        # Extract tables
        for table in doc.tables:
            table_data = {
                "caption": getattr(table, "caption", None),
                "num_rows": len(table.data) if hasattr(table, "data") else 0,
            }
            structure.tables.append(table_data)

        return structure

    def _create_chunks(
        self, doc, markdown_text: str, options: ParseOptions
    ) -> List[ContentChunk]:
        """Create chunks from docling document."""
        chunks = []
        chunk_size = options.chunk_size

        # If document has pages, chunk by page
        if hasattr(doc, "pages") and doc.pages:
            for page_num, page in enumerate(doc.pages, 1):
                page_text = str(page)
                if page_text.strip():
                    chunks.append(
                        ContentChunk(
                            index=len(chunks),
                            content=page_text,
                            start_char=0,  # Approximate
                            end_char=len(page_text),
                            metadata=ChunkMetadata(
                                start_page=page_num,
                                end_page=page_num,
                            ),
                        )
                    )
        else:
            # Fallback to text-based chunking
            paragraphs = markdown_text.split("\n\n")
            current_chunk = ""
            char_pos = 0
            chunk_idx = 0

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                if len(current_chunk) + len(para) > chunk_size:
                    if current_chunk:
                        chunks.append(
                            ContentChunk(
                                index=chunk_idx,
                                content=current_chunk.strip(),
                                start_char=char_pos,
                                end_char=char_pos + len(current_chunk),
                            )
                        )
                        chunk_idx += 1
                        char_pos += len(current_chunk)
                    current_chunk = para + "\n\n"
                else:
                    current_chunk += para + "\n\n"

            if current_chunk.strip():
                chunks.append(
                    ContentChunk(
                        index=chunk_idx,
                        content=current_chunk.strip(),
                        start_char=char_pos,
                        end_char=char_pos + len(current_chunk),
                    )
                )

        return chunks
