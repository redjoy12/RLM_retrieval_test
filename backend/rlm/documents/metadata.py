"""Metadata extraction utilities for documents."""

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from langdetect import detect, LangDetectException

from rlm.documents.models import DocumentFormat, DocumentMetadata


class MetadataExtractor:
    """Extract metadata from document files."""

    def __init__(self) -> None:
        """Initialize metadata extractor."""
        pass

    async def extract(
        self,
        file_path: Path,
        content: str,
        format_type: DocumentFormat,
    ) -> DocumentMetadata:
        """
        Extract metadata from document.

        Args:
            file_path: Path to the file
            content: Extracted text content
            format_type: Document format

        Returns:
            DocumentMetadata with extracted information
        """
        metadata = DocumentMetadata()

        # File system metadata
        metadata.extraction_metadata["file_name"] = file_path.name
        metadata.extraction_metadata["file_size"] = file_path.stat().st_size
        metadata.extraction_metadata["file_modified"] = datetime.fromtimestamp(
            file_path.stat().st_mtime
        )

        # Content statistics
        metadata.char_count = len(content)
        metadata.word_count = len(content.split())

        # Language detection
        metadata.language = self._detect_language(content)

        # Format-specific extraction
        if format_type == DocumentFormat.PDF:
            metadata = await self._extract_pdf_metadata(file_path, metadata)
        elif format_type == DocumentFormat.DOCX:
            metadata = await self._extract_docx_metadata(file_path, metadata)
        elif format_type == DocumentFormat.MARKDOWN:
            metadata = self._extract_markdown_metadata(content, metadata)
        elif format_type == DocumentFormat.HTML:
            metadata = self._extract_html_metadata(content, metadata)

        # Extract title from content if not already set
        if not metadata.title:
            metadata.title = self._extract_title_from_content(content)

        return metadata

    def _detect_language(self, content: str) -> Optional[str]:
        """Detect language from content."""
        try:
            # Use first 1000 chars for faster detection
            sample = content[:1000]
            return detect(sample)
        except LangDetectException:
            return None

    def _extract_title_from_content(self, content: str) -> Optional[str]:
        """Extract title from content."""
        lines = content.split("\n")[:20]  # Check first 20 lines

        for line in lines:
            line = line.strip()

            # Markdown header
            if line.startswith("# "):
                return line[2:].strip()

            # Underlined header
            if line and len(lines) > lines.index(line) + 1:
                next_line = lines[lines.index(line) + 1]
                if next_line.strip() and all(c == "=" for c in next_line.strip()):
                    return line

            # All caps short line (likely title)
            if line and len(line) < 100 and line.isupper() and len(line.split()) <= 10:
                return line

        return None

    async def _extract_pdf_metadata(
        self, file_path: Path, metadata: DocumentMetadata
    ) -> DocumentMetadata:
        """Extract PDF-specific metadata."""
        try:
            import fitz

            doc = fitz.open(file_path)
            pdf_meta = doc.metadata

            metadata.title = pdf_meta.get("title") or metadata.title
            metadata.author = pdf_meta.get("author")
            metadata.subject = pdf_meta.get("subject")
            metadata.page_count = len(doc)

            if pdf_meta.get("creationDate"):
                try:
                    # PDF dates are in format D:YYYYMMDDHHmmSS
                    date_str = pdf_meta["creationDate"]
                    if date_str.startswith("D:"):
                        date_str = date_str[2:14]
                        metadata.created_date = datetime.strptime(date_str, "%Y%m%d%H%M%S")
                except ValueError:
                    pass

            doc.close()
        except Exception:
            pass

        return metadata

    async def _extract_docx_metadata(
        self, file_path: Path, metadata: DocumentMetadata
    ) -> DocumentMetadata:
        """Extract DOCX-specific metadata."""
        try:
            from docx import Document

            doc = Document(file_path)
            core_props = doc.core_properties

            metadata.title = core_props.title or metadata.title
            metadata.author = core_props.author
            metadata.subject = core_props.subject
            metadata.created_date = core_props.created
            metadata.modified_date = core_props.modified

        except Exception:
            pass

        return metadata

    def _extract_markdown_metadata(
        self, content: str, metadata: DocumentMetadata
    ) -> DocumentMetadata:
        """Extract Markdown-specific metadata."""
        # Try to extract YAML frontmatter
        frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
        match = re.match(frontmatter_pattern, content, re.DOTALL)

        if match:
            try:
                import yaml

                frontmatter = yaml.safe_load(match.group(1))
                if isinstance(frontmatter, dict):
                    metadata.title = frontmatter.get("title", metadata.title)
                    metadata.author = frontmatter.get("author")
                    metadata.extraction_metadata["frontmatter"] = frontmatter
            except Exception:
                pass

        # Extract title from first header
        if not metadata.title:
            header_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
            if header_match:
                metadata.title = header_match.group(1)

        return metadata

    def _extract_html_metadata(self, content: str, metadata: DocumentMetadata) -> DocumentMetadata:
        """Extract HTML-specific metadata."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(content, "html.parser")

            # Extract title
            if soup.title and soup.title.string:
                metadata.title = soup.title.string.strip()

            # Extract meta tags
            meta_tags = {}
            for meta in soup.find_all("meta"):
                name = meta.get("name", "").lower()
                content_val = meta.get("content", "")
                if name and content_val:
                    meta_tags[name] = content_val

            if "author" in meta_tags:
                metadata.author = meta_tags["author"]
            if "description" in meta_tags:
                metadata.extraction_metadata["description"] = meta_tags["description"]
            if "keywords" in meta_tags:
                metadata.keywords = [k.strip() for k in meta_tags["keywords"].split(",")]

        except Exception:
            pass

        return metadata


class LLMMetadataExtractor:
    """Use LLM to extract metadata from documents."""

    def __init__(self, llm_client=None) -> None:
        """
        Initialize LLM metadata extractor.

        Args:
            llm_client: LLM client for metadata extraction
        """
        self.llm_client = llm_client

    async def extract(self, content: str, max_chars: int = 4000) -> Dict[str, Any]:
        """
        Extract metadata using LLM.

        Args:
            content: Document content
            max_chars: Maximum characters to send to LLM

        Returns:
            Dictionary of extracted metadata
        """
        if not self.llm_client:
            return {}

        # Use first part of content for extraction
        sample = content[:max_chars]

        prompt = f"""Extract the following metadata from this document sample:

Document Sample:
{sample}

---

Please provide:
1. Title (if any)
2. Author(s) (if any)
3. Main topic/subject
4. Key entities (people, organizations, locations)
5. Document type
6. Key dates mentioned

Format your response as JSON with these keys: title, authors, topic, entities, document_type, dates"""

        try:
            response = await self.llm_client.generate(prompt)

            # Try to parse JSON from response
            import json

            # Find JSON in response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                metadata = json.loads(json_match.group())
                return metadata

        except Exception:
            pass

        return {}
