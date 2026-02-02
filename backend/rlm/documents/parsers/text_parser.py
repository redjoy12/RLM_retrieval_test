"""Text and Markdown document parsers."""

import re
from pathlib import Path
from typing import List, Optional

import aiofiles
import yaml
from markdown import Markdown

from rlm.documents.models import (
    ContentChunk,
    DocumentContent,
    DocumentFormat,
    DocumentStructure,
    ParseOptions,
)
from rlm.documents.parsers.base import DocumentParser


class TextParser(DocumentParser):
    """Parser for plain text documents."""

    SUPPORTED_FORMATS = {DocumentFormat.PLAIN_TEXT}

    @property
    def name(self) -> str:
        return "text"

    @property
    def priority(self) -> int:
        return 100

    def supports(self, format_type: DocumentFormat) -> bool:
        return format_type in self.SUPPORTED_FORMATS

    async def parse(
        self, file_path: Path, options: ParseOptions
    ) -> DocumentContent:
        try:
            # Try different encodings
            content = await self._read_with_encoding(file_path)

            return DocumentContent(
                raw_text=content,
                cleaned_text=content,
                chunks=self._chunk_content(content, options),
                structure=DocumentStructure(),
            )
        except Exception as e:
            raise ParseError(f"Failed to parse text file: {e}") from e

    async def _read_with_encoding(self, file_path: Path) -> str:
        """Read file trying multiple encodings."""
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

        for encoding in encodings:
            try:
                async with aiofiles.open(
                    file_path, "r", encoding=encoding
                ) as f:
                    return await f.read()
            except UnicodeDecodeError:
                continue

        raise ParseError(f"Could not decode file with any encoding: {file_path}")

    def _chunk_content(
        self, content: str, options: ParseOptions
    ) -> List[ContentChunk]:
        """Split content into chunks."""
        chunks = []
        chunk_size = options.chunk_size
        overlap = options.chunk_overlap

        # Try to split at paragraph boundaries
        paragraphs = content.split("\n\n")
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
                        )
                    )
                    chunk_idx += 1
                    char_pos += len(current_chunk) - overlap

                # Start new chunk with overlap
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
                )
            )

        return chunks


class MarkdownParser(DocumentParser):
    """Parser for Markdown documents."""

    SUPPORTED_FORMATS = {DocumentFormat.MARKDOWN}

    @property
    def name(self) -> str:
        return "markdown"

    @property
    def priority(self) -> int:
        return 100

    def supports(self, format_type: DocumentFormat) -> bool:
        return format_type in self.SUPPORTED_FORMATS

    async def parse(
        self, file_path: Path, options: ParseOptions
    ) -> DocumentContent:
        try:
            # Read file
            content = await self._read_with_encoding(file_path)

            # Extract frontmatter
            frontmatter, body = self._extract_frontmatter(content)

            # Parse structure
            structure = self._parse_structure(body)

            # Convert to HTML if needed for cleaning
            if not options.preserve_formatting:
                md = Markdown()
                body = md.convert(body)

            return DocumentContent(
                raw_text=content,
                cleaned_text=body,
                chunks=self._chunk_content(body, options),
                structure=structure,
            )
        except Exception as e:
            raise ParseError(f"Failed to parse markdown file: {e}") from e

    async def _read_with_encoding(self, file_path: Path) -> str:
        """Read file trying multiple encodings."""
        encodings = ["utf-8", "utf-8-sig", "latin-1"]

        for encoding in encodings:
            try:
                async with aiofiles.open(
                    file_path, "r", encoding=encoding
                ) as f:
                    return await f.read()
            except UnicodeDecodeError:
                continue

        raise ParseError(f"Could not decode file: {file_path}")

    def _extract_frontmatter(self, content: str) -> tuple:
        """Extract YAML frontmatter from markdown."""
        pattern = r"^---\s*\n(.*?)\n---\s*\n"
        match = re.match(pattern, content, re.DOTALL)

        if match:
            try:
                frontmatter = yaml.safe_load(match.group(1))
                body = content[match.end() :]
                return frontmatter or {}, body
            except yaml.YAMLError:
                pass

        return {}, content

    def _parse_structure(self, content: str) -> DocumentStructure:
        """Parse markdown structure (headings)."""
        structure = DocumentStructure()

        # Extract headings
        heading_pattern = r"^(#{1,6})\s+(.+)$"
        for line in content.split("\n"):
            match = re.match(heading_pattern, line.strip())
            if match:
                level = len(match.group(1))
                title = match.group(2)
                structure.headings.append(f"{'#' * level} {title}")

        # Extract first heading as title
        if structure.headings:
            structure.title = structure.headings[0].lstrip("# ")

        return structure

    def _chunk_content(
        self, content: str, options: ParseOptions
    ) -> List[ContentChunk]:
        """Split markdown content into chunks at heading boundaries."""
        chunks = []
        chunk_size = options.chunk_size
        overlap = options.chunk_overlap

        # Split by headers
        sections = self._split_by_headers(content)
        current_chunk = ""
        char_pos = 0
        chunk_idx = 0
        current_section = ""

        for section_title, section_content in sections:
            section_text = f"{section_title}\n{section_content}\n\n"

            if len(current_chunk) + len(section_text) > chunk_size:
                # Save current chunk
                if current_chunk:
                    chunks.append(
                        ContentChunk(
                            index=chunk_idx,
                            content=current_chunk.strip(),
                            start_char=char_pos,
                            end_char=char_pos + len(current_chunk),
                            metadata={"section_title": current_section},
                        )
                    )
                    chunk_idx += 1
                    char_pos += len(current_chunk) - overlap

                # Start new chunk
                if overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-overlap:]
                    current_chunk = overlap_text + section_text
                else:
                    current_chunk = section_text
                current_section = section_title
            else:
                current_chunk += section_text

        # Add final chunk
        if current_chunk.strip():
            chunks.append(
                ContentChunk(
                    index=chunk_idx,
                    content=current_chunk.strip(),
                    start_char=char_pos,
                    end_char=char_pos + len(current_chunk),
                    metadata={"section_title": current_section},
                )
            )

        return chunks

    def _split_by_headers(self, content: str) -> List[tuple]:
        """Split content by markdown headers."""
        sections = []
        lines = content.split("\n")
        current_title = ""
        current_content = []

        header_pattern = r"^(#{1,6})\s+(.+)$"

        for line in lines:
            match = re.match(header_pattern, line.strip())
            if match:
                # Save previous section
                if current_content:
                    sections.append(
                        (current_title, "\n".join(current_content))
                    )
                current_title = line.strip()
                current_content = []
            else:
                current_content.append(line)

        # Add final section
        if current_content or current_title:
            sections.append((current_title, "\n".join(current_content)))

        # If no headers found, treat entire content as one section
        if not sections:
            sections.append(("", content))

        return sections


class ParseError(Exception):
    """Error during document parsing."""

    pass
