"""Content chunking engine for document processing."""

from typing import List, Optional

from rlm.documents.models import ChunkMetadata, ContentChunk, ParseOptions


class ChunkingEngine:
    """Engine for chunking document content."""

    # Average tokens per character (rough estimate)
    CHARS_PER_TOKEN = 4

    def __init__(
        self,
        chunk_size: int = 100000,
        chunk_overlap: int = 1000,
        respect_boundaries: bool = True,
    ) -> None:
        """
        Initialize chunking engine.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            respect_boundaries: Try to respect paragraph/section boundaries
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_boundaries = respect_boundaries

    def chunk(
        self, text: str, options: Optional[ParseOptions] = None
    ) -> List[ContentChunk]:
        """
        Split text into chunks.

        Args:
            text: Text to chunk
            options: Parse options (overrides engine settings)

        Returns:
            List of content chunks
        """
        if options:
            chunk_size = options.chunk_size
            overlap = options.chunk_overlap
        else:
            chunk_size = self.chunk_size
            overlap = self.chunk_overlap

        if not text:
            return []

        # If text is small enough, return as single chunk
        if len(text) <= chunk_size:
            return [
                ContentChunk(
                    index=0,
                    content=text,
                    start_char=0,
                    end_char=len(text),
                    token_count=self._estimate_tokens(text),
                )
            ]

        if self.respect_boundaries:
            return self._chunk_with_boundaries(text, chunk_size, overlap)
        else:
            return self._chunk_fixed_size(text, chunk_size, overlap)

    def _chunk_with_boundaries(
        self, text: str, chunk_size: int, overlap: int
    ) -> List[ContentChunk]:
        """Chunk text respecting paragraph boundaries."""
        chunks = []

        # Split by paragraphs first
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
                # Save current chunk
                if current_chunk:
                    chunks.append(
                        ContentChunk(
                            index=chunk_idx,
                            content=current_chunk.strip(),
                            start_char=char_pos,
                            end_char=char_pos + len(current_chunk),
                            token_count=self._estimate_tokens(current_chunk),
                        )
                    )
                    chunk_idx += 1
                    char_pos += len(current_chunk) - overlap

                # Handle overlap
                if overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-overlap:]
                    current_chunk = overlap_text + para_with_newline
                else:
                    current_chunk = para_with_newline
            else:
                current_chunk += para_with_newline

        # Add final chunk
        if current_chunk.strip():
            chunks.append(
                ContentChunk(
                    index=chunk_idx,
                    content=current_chunk.strip(),
                    start_char=char_pos,
                    end_char=char_pos + len(current_chunk),
                    token_count=self._estimate_tokens(current_chunk),
                )
            )

        return chunks

    def _chunk_fixed_size(
        self, text: str, chunk_size: int, overlap: int
    ) -> List[ContentChunk]:
        """Chunk text with fixed size."""
        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            # Try to break at word boundary
            if end < len(text):
                # Look for space or newline
                while end > start and text[end - 1] not in " \n":
                    end -= 1
                if end == start:
                    end = min(start + chunk_size, len(text))

            chunk_text = text[start:end]

            chunks.append(
                ContentChunk(
                    index=chunk_idx,
                    content=chunk_text,
                    start_char=start,
                    end_char=end,
                    token_count=self._estimate_tokens(chunk_text),
                )
            )

            chunk_idx += 1
            start = end - overlap

        return chunks

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from character count."""
        return len(text) // self.CHARS_PER_TOKEN

    def rechunk_existing(
        self, chunks: List[ContentChunk], target_size: int
    ) -> List[ContentChunk]:
        """
        Rechunk existing chunks to target size.

        Args:
            chunks: Existing chunks
            target_size: Target chunk size

        Returns:
            Rechunked content
        """
        # Combine all chunks
        full_text = "\n\n".join(chunk.content for chunk in chunks)

        # Rechunk
        new_chunks = self._chunk_with_boundaries(
            full_text, target_size, self.chunk_overlap
        )

        # Preserve metadata from original chunks where possible
        for new_chunk in new_chunks:
            # Find overlapping original chunks
            overlapping = [
                chunk
                for chunk in chunks
                if not (
                    chunk.end_char <= new_chunk.start_char
                    or chunk.start_char >= new_chunk.end_char
                )
            ]

            if overlapping:
                # Merge metadata from overlapping chunks
                new_chunk.metadata = self._merge_metadata(
                    [chunk.metadata for chunk in overlapping]
                )

        return new_chunks

    def _merge_metadata(self, metadata_list: List[ChunkMetadata]) -> ChunkMetadata:
        """Merge metadata from multiple chunks."""
        if not metadata_list:
            return ChunkMetadata()

        # Take first chunk's metadata as base
        merged = metadata_list[0]

        # Combine page ranges
        all_pages = set()
        for meta in metadata_list:
            if meta.start_page:
                all_pages.add(meta.start_page)
            if meta.end_page:
                all_pages.add(meta.end_page)

        if all_pages:
            merged.start_page = min(all_pages)
            merged.end_page = max(all_pages)

        # If any chunk has special flags, preserve them
        merged.is_table = any(meta.is_table for meta in metadata_list)
        merged.is_code = any(meta.is_code for meta in metadata_list)

        return merged


class SemanticChunker:
    """Chunk content based on semantic boundaries."""

    def __init__(
        self,
        chunk_size: int = 100000,
        chunk_overlap: int = 1000,
    ) -> None:
        """
        Initialize semantic chunker.

        Args:
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self, text: str, headings: Optional[List[str]] = None
    ) -> List[ContentChunk]:
        """
        Chunk text based on semantic boundaries.

        Args:
            text: Text to chunk
            headings: List of section headings for boundary detection

        Returns:
            List of content chunks
        """
        if not text:
            return []

        # If we have headings, use them as boundaries
        if headings:
            return self._chunk_by_headings(text, headings)

        # Otherwise fall back to paragraph chunking
        engine = ChunkingEngine(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            respect_boundaries=True,
        )
        return engine.chunk(text)

    def _chunk_by_headings(
        self, text: str, headings: List[str]
    ) -> List[ContentChunk]:
        """Chunk text by section headings."""
        chunks = []
        char_pos = 0
        chunk_idx = 0

        # Split by headings
        sections = self._split_by_headings(text, headings)

        current_chunk_text = ""
        current_chunk_sections = []

        for section_title, section_text in sections:
            section_full = f"{section_title}\n{section_text}\n\n"

            if (
                len(current_chunk_text) + len(section_full) > self.chunk_size
                and current_chunk_text
            ):
                # Save current chunk
                chunks.append(
                    ContentChunk(
                        index=chunk_idx,
                        content=current_chunk_text.strip(),
                        start_char=char_pos,
                        end_char=char_pos + len(current_chunk_text),
                        metadata=ChunkMetadata(
                            section_title=current_chunk_sections[0]
                            if current_chunk_sections
                            else ""
                        ),
                    )
                )
                chunk_idx += 1
                char_pos += len(current_chunk_text) - self.chunk_overlap

                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk_text:
                    overlap_text = current_chunk_text[-self.chunk_overlap :]
                    current_chunk_text = overlap_text + section_full
                else:
                    current_chunk_text = section_full
                current_chunk_sections = [section_title]
            else:
                current_chunk_text += section_full
                current_chunk_sections.append(section_title)

        # Add final chunk
        if current_chunk_text.strip():
            chunks.append(
                ContentChunk(
                    index=chunk_idx,
                    content=current_chunk_text.strip(),
                    start_char=char_pos,
                    end_char=char_pos + len(current_chunk_text),
                    metadata=ChunkMetadata(
                        section_title=current_chunk_sections[0]
                        if current_chunk_sections
                        else ""
                    ),
                )
            )

        return chunks

    def _split_by_headings(
        self, text: str, headings: List[str]
    ) -> List[tuple]:
        """Split text into sections by headings."""
        sections = []
        remaining_text = text

        for heading in headings:
            # Find heading in text
            heading_pos = remaining_text.find(heading)
            if heading_pos == -1:
                continue

            # Content before this heading goes to previous section
            if heading_pos > 0 and sections:
                sections[-1] = (
                    sections[-1][0],
                    sections[-1][1] + remaining_text[:heading_pos],
                )
            elif heading_pos > 0 and not sections:
                # Text before first heading
                sections.append(("", remaining_text[:heading_pos]))

            # Find next heading
            next_heading_pos = len(remaining_text)
            for next_heading in headings:
                if next_heading == heading:
                    continue
                pos = remaining_text.find(next_heading, heading_pos + len(heading))
                if pos != -1 and pos < next_heading_pos:
                    next_heading_pos = pos

            section_content = remaining_text[
                heading_pos + len(heading) : next_heading_pos
            ]
            sections.append((heading, section_content))

            remaining_text = remaining_text[next_heading_pos:]

        # Add remaining text to last section
        if remaining_text and sections:
            sections[-1] = (
                sections[-1][0],
                sections[-1][1] + remaining_text,
            )
        elif remaining_text and not sections:
            sections.append(("", remaining_text))

        return sections
