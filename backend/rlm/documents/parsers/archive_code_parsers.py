"""Archive and code file parsers."""

import asyncio
import io
import tarfile
import zipfile
from pathlib import Path
from typing import List, Set

import aiofiles

from rlm.documents.models import (
    ContentChunk,
    DocumentContent,
    DocumentFormat,
    DocumentStructure,
    ParseOptions,
)
from rlm.documents.parsers.base import DocumentParser
from rlm.documents.parsers.text_parser import ParseError


class ArchiveParser(DocumentParser):
    """Parser for archive files (ZIP, TAR)."""

    SUPPORTED_FORMATS = {DocumentFormat.ZIP, DocumentFormat.TAR}

    # Files to skip in archives
    SKIP_EXTENSIONS = {
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".bin",
        ".o",
        ".a",
    }

    @property
    def name(self) -> str:
        return "archive"

    @property
    def priority(self) -> int:
        return 100

    def supports(self, format_type: DocumentFormat) -> bool:
        return format_type in self.SUPPORTED_FORMATS

    async def parse(
        self, file_path: Path, options: ParseOptions
    ) -> DocumentContent:
        try:
            format_type = self._detect_format(file_path)

            if format_type == DocumentFormat.ZIP:
                return await self._parse_zip(file_path, options)
            elif format_type == DocumentFormat.TAR:
                return await self._parse_tar(file_path, options)
            else:
                raise ParseError(f"Unsupported archive format: {format_type}")

        except Exception as e:
            raise ParseError(f"Failed to parse archive: {e}") from e

    def _detect_format(self, file_path: Path) -> DocumentFormat:
        """Detect archive format from file."""
        suffix = file_path.suffix.lower()

        if suffix == ".zip":
            return DocumentFormat.ZIP
        elif suffix in (".tar", ".gz", ".tgz", ".bz2"):
            return DocumentFormat.TAR

        # Try to detect by content
        with open(file_path, "rb") as f:
            header = f.read(4)
            if header[:2] == b"PK":
                return DocumentFormat.ZIP

        return DocumentFormat.UNKNOWN

    async def _parse_zip(
        self, file_path: Path, options: ParseOptions
    ) -> DocumentContent:
        """Parse ZIP archive."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._parse_zip_sync, file_path, options
        )

    def _parse_zip_sync(
        self, file_path: Path, options: ParseOptions
    ) -> DocumentContent:
        """Synchronous ZIP parsing."""
        text_parts = ["[ARCHIVE CONTENTS]\n"]

        with zipfile.ZipFile(file_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue

                file_name = info.filename
                ext = Path(file_name).suffix.lower()

                if ext in self.SKIP_EXTENSIONS:
                    continue

                # Try to read as text
                try:
                    with zf.open(info) as f:
                        content = f.read()
                        try:
                            text = content.decode("utf-8")
                            text_parts.append(f"\n--- {file_name} ---\n")
                            text_parts.append(text)
                        except UnicodeDecodeError:
                            # Binary file, skip
                            pass
                except Exception:
                    pass

        full_text = "\n".join(text_parts)

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

    async def _parse_tar(
        self, file_path: Path, options: ParseOptions
    ) -> DocumentContent:
        """Parse TAR archive."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._parse_tar_sync, file_path, options
        )

    def _parse_tar_sync(
        self, file_path: Path, options: ParseOptions
    ) -> DocumentContent:
        """Synchronous TAR parsing."""
        text_parts = ["[ARCHIVE CONTENTS]\n"]

        with tarfile.open(file_path, "r:*") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue

                file_name = member.name
                ext = Path(file_name).suffix.lower()

                if ext in self.SKIP_EXTENSIONS:
                    continue

                # Try to read as text
                try:
                    f = tf.extractfile(member)
                    if f:
                        content = f.read()
                        try:
                            text = content.decode("utf-8")
                            text_parts.append(f"\n--- {file_name} ---\n")
                            text_parts.append(text)
                        except UnicodeDecodeError:
                            pass
                except Exception:
                    pass

        full_text = "\n".join(text_parts)

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


class CodeParser(DocumentParser):
    """Parser for source code files."""

    SUPPORTED_FORMATS = {
        DocumentFormat.PYTHON,
        DocumentFormat.JAVASCRIPT,
        DocumentFormat.TYPESCRIPT,
        DocumentFormat.JAVA,
        DocumentFormat.CPP,
        DocumentFormat.C,
        DocumentFormat.GO,
        DocumentFormat.RUST,
        DocumentFormat.RUBY,
        DocumentFormat.PHP,
    }

    # Language display names
    LANGUAGE_NAMES = {
        DocumentFormat.PYTHON: "Python",
        DocumentFormat.JAVASCRIPT: "JavaScript",
        DocumentFormat.TYPESCRIPT: "TypeScript",
        DocumentFormat.JAVA: "Java",
        DocumentFormat.CPP: "C++",
        DocumentFormat.C: "C",
        DocumentFormat.GO: "Go",
        DocumentFormat.RUST: "Rust",
        DocumentFormat.RUBY: "Ruby",
        DocumentFormat.PHP: "PHP",
    }

    @property
    def name(self) -> str:
        return "code"

    @property
    def priority(self) -> int:
        return 100

    def supports(self, format_type: DocumentFormat) -> bool:
        return format_type in self.SUPPORTED_FORMATS

    async def parse(
        self, file_path: Path, options: ParseOptions
    ) -> DocumentContent:
        try:
            # Detect format from extension
            format_type = DocumentFormat.from_extension(file_path.suffix)

            if not self.supports(format_type):
                raise ParseError(f"Unsupported code format: {format_type}")

            # Read file
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()

            # Add language identifier
            lang_name = self.LANGUAGE_NAMES.get(format_type, "Code")
            formatted_text = f"[{lang_name} CODE]\n\n{content}"

            chunks = [
                ContentChunk(
                    index=0,
                    content=formatted_text,
                    start_char=0,
                    end_char=len(formatted_text),
                    metadata={"language": lang_name.lower(), "is_code": True},
                )
            ]

            return DocumentContent(
                raw_text=content,
                cleaned_text=formatted_text,
                chunks=chunks,
                structure=DocumentStructure(),
            )

        except Exception as e:
            raise ParseError(f"Failed to parse code file: {e}") from e
