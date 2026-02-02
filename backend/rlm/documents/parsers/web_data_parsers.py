"""Web and data format parsers (HTML, CSV, JSON, XML)."""

import asyncio
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List

import aiofiles
import pandas as pd
from bs4 import BeautifulSoup

from rlm.documents.models import (
    ContentChunk,
    DocumentContent,
    DocumentFormat,
    DocumentStructure,
    ParseOptions,
)
from rlm.documents.parsers.base import DocumentParser
from rlm.documents.parsers.text_parser import ParseError


class HtmlParser(DocumentParser):
    """Parser for HTML documents."""

    SUPPORTED_FORMATS = {DocumentFormat.HTML}

    @property
    def name(self) -> str:
        return "html"

    @property
    def priority(self) -> int:
        return 100

    def supports(self, format_type: DocumentFormat) -> bool:
        return format_type in self.SUPPORTED_FORMATS

    async def parse(
        self, file_path: Path, options: ParseOptions
    ) -> DocumentContent:
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()

            soup = BeautifulSoup(content, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Extract title
            title = ""
            if soup.title:
                title = soup.title.get_text()

            # Extract text
            text = soup.get_text()

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks_text = (phrase.strip() for line in lines for phrase in line.split("  "))
            cleaned_text = "\n".join(chunk for chunk in chunks_text if chunk)

            # Extract structure
            structure = DocumentStructure(
                title=title,
                headings=[h.get_text() for h in soup.find_all(["h1", "h2", "h3"])],
            )

            # Create chunks
            chunks = self._chunk_content(cleaned_text, options)

            return DocumentContent(
                raw_text=content,
                cleaned_text=cleaned_text,
                chunks=chunks,
                structure=structure,
            )

        except Exception as e:
            raise ParseError(f"Failed to parse HTML: {e}") from e

    def _chunk_content(self, text: str, options: ParseOptions) -> List[ContentChunk]:
        """Split content into chunks."""
        chunks = []
        paragraphs = text.split("\n")
        current_chunk = ""
        char_pos = 0
        chunk_idx = 0
        chunk_size = options.chunk_size

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_with_newline = para + "\n"

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
                    char_pos += len(current_chunk)
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


class CsvParser(DocumentParser):
    """Parser for CSV files."""

    SUPPORTED_FORMATS = {DocumentFormat.CSV}

    @property
    def name(self) -> str:
        return "csv"

    @property
    def priority(self) -> int:
        return 100

    def supports(self, format_type: DocumentFormat) -> bool:
        return format_type in self.SUPPORTED_FORMATS

    async def parse(
        self, file_path: Path, options: ParseOptions
    ) -> DocumentContent:
        try:
            # Try to detect encoding and dialect
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None, pd.read_csv, file_path
            )

            # Convert to text representation
            text_parts = []

            # Add header
            headers = " | ".join(str(col) for col in df.columns)
            text_parts.append(f"Headers: {headers}")
            text_parts.append("")

            # Add rows
            for idx, row in df.iterrows():
                row_text = " | ".join(str(val) for val in row.values)
                text_parts.append(row_text)

            full_text = "\n".join(text_parts)

            # Single chunk for CSV
            chunks = [
                ContentChunk(
                    index=0,
                    content=full_text,
                    start_char=0,
                    end_char=len(full_text),
                )
            ]

            return DocumentContent(
                raw_text=full_text,
                cleaned_text=full_text,
                chunks=chunks,
                structure=DocumentStructure(),
            )

        except Exception as e:
            raise ParseError(f"Failed to parse CSV: {e}") from e


class JsonParser(DocumentParser):
    """Parser for JSON files."""

    SUPPORTED_FORMATS = {DocumentFormat.JSON}

    @property
    def name(self) -> str:
        return "json"

    @property
    def priority(self) -> int:
        return 100

    def supports(self, format_type: DocumentFormat) -> bool:
        return format_type in self.SUPPORTED_FORMATS

    async def parse(
        self, file_path: Path, options: ParseOptions
    ) -> DocumentContent:
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()

            data = json.loads(content)

            # Convert to readable text
            text = self._json_to_text(data)

            chunks = [
                ContentChunk(
                    index=0,
                    content=text,
                    start_char=0,
                    end_char=len(text),
                )
            ]

            return DocumentContent(
                raw_text=content,
                cleaned_text=text,
                chunks=chunks,
                structure=DocumentStructure(),
            )

        except Exception as e:
            raise ParseError(f"Failed to parse JSON: {e}") from e

    def _json_to_text(self, data: Any, indent: int = 0) -> str:
        """Convert JSON data to readable text."""
        lines = []
        prefix = "  " * indent

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.append(self._json_to_text(value, indent + 1))
                else:
                    lines.append(f"{prefix}{key}: {value}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}[{i}]:")
                    lines.append(self._json_to_text(item, indent + 1))
                else:
                    lines.append(f"{prefix}[{i}]: {item}")
        else:
            lines.append(f"{prefix}{data}")

        return "\n".join(lines)


class XmlParser(DocumentParser):
    """Parser for XML files."""

    SUPPORTED_FORMATS = {DocumentFormat.XML}

    @property
    def name(self) -> str:
        return "xml"

    @property
    def priority(self) -> int:
        return 100

    def supports(self, format_type: DocumentFormat) -> bool:
        return format_type in self.SUPPORTED_FORMATS

    async def parse(
        self, file_path: Path, options: ParseOptions
    ) -> DocumentContent:
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()

            root = ET.fromstring(content)
            text = self._xml_to_text(root)

            chunks = [
                ContentChunk(
                    index=0,
                    content=text,
                    start_char=0,
                    end_char=len(text),
                )
            ]

            return DocumentContent(
                raw_text=content,
                cleaned_text=text,
                chunks=chunks,
                structure=DocumentStructure(),
            )

        except Exception as e:
            raise ParseError(f"Failed to parse XML: {e}") from e

    def _xml_to_text(self, element: ET.Element, indent: int = 0) -> str:
        """Convert XML element to readable text."""
        lines = []
        prefix = "  " * indent

        # Add tag name
        tag_text = f"{prefix}{element.tag}"

        # Add attributes
        if element.attrib:
            attrs = ", ".join(f"{k}={v}" for k, v in element.attrib.items())
            tag_text += f" ({attrs})"

        # Add text content if present
        if element.text and element.text.strip():
            tag_text += f": {element.text.strip()}"

        lines.append(tag_text)

        # Process children
        for child in element:
            lines.append(self._xml_to_text(child, indent + 1))

        return "\n".join(lines)
