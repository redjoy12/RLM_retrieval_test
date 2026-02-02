"""Unified document parser using unstructured.io library."""

import asyncio
from pathlib import Path
from typing import List, Optional

from rlm.documents.models import (
    ChunkMetadata,
    ContentChunk,
    DocumentContent,
    DocumentFormat,
    DocumentStructure,
    ParseOptions,
)
from rlm.documents.parsers.base import DocumentParser
from rlm.documents.parsers.text_parser import ParseError


class UnstructuredParser(DocumentParser):
    """
    Unified parser using unstructured.io library.
    
    Supports 20+ formats with intelligent content extraction.
    Falls back to other parsers if unstructured is not available.
    """

    SUPPORTED_FORMATS = {
        DocumentFormat.PLAIN_TEXT,
        DocumentFormat.MARKDOWN,
        DocumentFormat.PDF,
        DocumentFormat.DOCX,
        DocumentFormat.XLSX,
        DocumentFormat.PPTX,
        DocumentFormat.HTML,
        DocumentFormat.XML,
        DocumentFormat.CSV,
        DocumentFormat.JSON,
        DocumentFormat.ZIP,
        DocumentFormat.EPUB,
        DocumentFormat.RST,
        DocumentFormat.ORG,
        DocumentFormat.JUPYTER,
    }

    def __init__(self) -> None:
        """Initialize unstructured parser."""
        self._unstructured_available = False
        self._partition = None
        
        try:
            from unstructured.partition.auto import partition
            self._partition = partition
            self._unstructured_available = True
        except ImportError:
            self._partition = None

    @property
    def name(self) -> str:
        return "unstructured"

    @property
    def priority(self) -> int:
        return 90  # High priority but not higher than specialized parsers

    def supports(self, format_type: DocumentFormat) -> bool:
        if not self._unstructured_available:
            return False
        return format_type in self.SUPPORTED_FORMATS

    async def parse(
        self, file_path: Path, options: ParseOptions
    ) -> DocumentContent:
        """
        Parse document using unstructured.
        
        Args:
            file_path: Path to document
            options: Parse options
            
        Returns:
            DocumentContent with extracted text and chunks
            
        Raises:
            ParseError: If parsing fails
        """
        if not self._unstructured_available:
            raise ParseError("unstructured library is not installed")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._parse_sync, file_path, options
        )

    def _parse_sync(self, file_path: Path, options: ParseOptions) -> DocumentContent:
        """Synchronous parsing using unstructured."""
        try:
            # Partition document into elements
            elements = self._partition(
                filename=str(file_path),
                include_page_breaks=True,
            )

            # Extract text from elements
            texts = []
            structure = DocumentStructure()
            current_page = 1
            
            for element in elements:
                text = str(element)
                if not text.strip():
                    continue
                
                texts.append(text)
                
                # Extract metadata from element
                metadata = element.metadata.to_dict() if hasattr(element, 'metadata') else {}
                
                # Update structure
                element_type = type(element).__name__
                if element_type in ['Title', 'Subject']:
                    structure.headings.append(text)
                    if not structure.title:
                        structure.title = text
                elif element_type == 'Header':
                    structure.headings.append(text)
                elif element_type == 'Table':
                    structure.tables.append({'content': text})
                
                # Track page numbers
                page_number = metadata.get('page_number', current_page)
                if page_number:
                    current_page = page_number

            full_text = "\n\n".join(texts)
            
            # Create chunks based on elements
            chunks = self._create_chunks_from_elements(
                elements, full_text, options
            )

            return DocumentContent(
                raw_text=full_text,
                cleaned_text=full_text,
                chunks=chunks,
                structure=structure,
            )

        except Exception as e:
            raise ParseError(f"Failed to parse with unstructured: {e}") from e

    def _create_chunks_from_elements(
        self, elements, full_text: str, options: ParseOptions
    ) -> List[ContentChunk]:
        """Create content chunks from unstructured elements."""
        chunks = []
        chunk_size = options.chunk_size
        overlap = options.chunk_overlap
        
        current_chunk_texts = []
        current_length = 0
        char_pos = 0
        chunk_idx = 0
        current_page = 1
        
        for element in elements:
            text = str(element)
            if not text.strip():
                continue
            
            # Get metadata
            metadata = element.metadata.to_dict() if hasattr(element, 'metadata') else {}
            page_number = metadata.get('page_number', current_page)
            element_type = type(element).__name__
            
            text_with_newline = text + "\n\n"
            text_length = len(text_with_newline)
            
            # Check if we need to start a new chunk
            if current_length + text_length > chunk_size and current_chunk_texts:
                # Save current chunk
                chunk_content = "".join(current_chunk_texts).strip()
                chunks.append(
                    ContentChunk(
                        index=chunk_idx,
                        content=chunk_content,
                        start_char=char_pos,
                        end_char=char_pos + len(chunk_content),
                        metadata=ChunkMetadata(
                            start_page=current_page,
                            end_page=page_number,
                            is_table=element_type == 'Table',
                            is_code=element_type == 'CodeSnippet',
                            custom={'element_types': list(set(
                                type(e).__name__ for e in elements[chunks[-1].index:chunk_idx+1]
                            ))} if chunks else {},
                        ),
                    )
                )
                chunk_idx += 1
                char_pos += len(chunk_content) - overlap
                
                # Handle overlap
                if overlap > 0 and len(chunk_content) > overlap:
                    overlap_text = chunk_content[-overlap:]
                    current_chunk_texts = [overlap_text + text_with_newline]
                    current_length = len(overlap_text) + text_length
                else:
                    current_chunk_texts = [text_with_newline]
                    current_length = text_length
            else:
                current_chunk_texts.append(text_with_newline)
                current_length += text_length
            
            current_page = page_number
        
        # Add final chunk
        if current_chunk_texts:
            chunk_content = "".join(current_chunk_texts).strip()
            chunks.append(
                ContentChunk(
                    index=chunk_idx,
                    content=chunk_content,
                    start_char=char_pos,
                    end_char=char_pos + len(chunk_content),
                    metadata=ChunkMetadata(
                        start_page=current_page,
                        end_page=current_page,
                    ),
                )
            )
        
        return chunks

    def _chunk_content_fallback(
        self, text: str, options: ParseOptions
    ) -> List[ContentChunk]:
        """Fallback chunking when element-based chunking fails."""
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
