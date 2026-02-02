"""Format detection for document files."""

import mimetypes
from pathlib import Path
from typing import Optional, Tuple

import filetype
import magic

from rlm.documents.models import DocumentFormat


class FormatDetector:
    """Detect document format from file content and extension."""

    # MIME type to format mapping
    MIME_MAPPING = {
        # Text formats
        "text/plain": DocumentFormat.PLAIN_TEXT,
        "text/markdown": DocumentFormat.MARKDOWN,
        "text/x-markdown": DocumentFormat.MARKDOWN,
        # Document formats
        "application/pdf": DocumentFormat.PDF,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocumentFormat.DOCX,
        "application/msword": DocumentFormat.DOCX,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": DocumentFormat.XLSX,
        "application/vnd.ms-excel": DocumentFormat.XLSX,
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": DocumentFormat.PPTX,
        "application/vnd.ms-powerpoint": DocumentFormat.PPTX,
        # Web formats
        "text/html": DocumentFormat.HTML,
        "application/xhtml+xml": DocumentFormat.HTML,
        "application/xml": DocumentFormat.XML,
        "text/xml": DocumentFormat.XML,
        # Data formats
        "text/csv": DocumentFormat.CSV,
        "application/json": DocumentFormat.JSON,
        "application/x-yaml": DocumentFormat.YAML,
        "text/yaml": DocumentFormat.YAML,
        # Archive formats
        "application/zip": DocumentFormat.ZIP,
        "application/x-tar": DocumentFormat.TAR,
        "application/gzip": DocumentFormat.GZIP,
        "application/x-gzip": DocumentFormat.GZIP,
        # Jupyter
        "application/x-ipynb+json": DocumentFormat.JUPYTER,
    }

    # Code file extensions
    CODE_EXTENSIONS = {
        ".py": DocumentFormat.PYTHON,
        ".js": DocumentFormat.JAVASCRIPT,
        ".ts": DocumentFormat.TYPESCRIPT,
        ".java": DocumentFormat.JAVA,
        ".cpp": DocumentFormat.CPP,
        ".c": DocumentFormat.C,
        ".go": DocumentFormat.GO,
        ".rs": DocumentFormat.RUST,
        ".rb": DocumentFormat.RUBY,
        ".php": DocumentFormat.PHP,
    }

    def __init__(self) -> None:
        """Initialize the format detector."""
        self._magic = None
        try:
            self._magic = magic.Magic(mime=True)
        except Exception:
            # magic library might not be available on all systems
            self._magic = None

    def detect(
        self, file_path: Path, use_content: bool = True
    ) -> Tuple[DocumentFormat, str]:
        """
        Detect document format from file.

        Args:
            file_path: Path to the file
            use_content: Whether to analyze file content (slower but more accurate)

        Returns:
            Tuple of (format, mime_type)
        """
        # First try extension-based detection
        ext_format, ext_mime = self._detect_from_extension(file_path)

        if not use_content:
            return ext_format, ext_mime

        # Try content-based detection for better accuracy
        try:
            content_format, content_mime = self._detect_from_content(file_path)

            # If content detection gives a specific result, use it
            if content_format != DocumentFormat.UNKNOWN:
                # Special handling for ZIP - could be many things
                if content_format == DocumentFormat.ZIP:
                    # Check if it's a docx/xlsx/pptx by extension
                    if ext_format in (DocumentFormat.DOCX, DocumentFormat.XLSX, DocumentFormat.PPTX):
                        return ext_format, ext_mime
                return content_format, content_mime
        except Exception:
            # Fall back to extension-based if content detection fails
            pass

        return ext_format, ext_mime

    def _detect_from_extension(self, file_path: Path) -> Tuple[DocumentFormat, str]:
        """Detect format from file extension."""
        suffix = file_path.suffix.lower()

        # Check code extensions first
        if suffix in self.CODE_EXTENSIONS:
            return self.CODE_EXTENSIONS[suffix], "text/plain"

        # Try to get MIME type from extension
        mime_type, _ = mimetypes.guess_type(str(file_path))

        if mime_type:
            format_type = self.MIME_MAPPING.get(mime_type, DocumentFormat.UNKNOWN)
            if format_type != DocumentFormat.UNKNOWN:
                return format_type, mime_type

        # Try DocumentFormat enum mapping
        format_type = DocumentFormat.from_extension(suffix)
        if format_type != DocumentFormat.UNKNOWN:
            # Get a reasonable MIME type
            mime_type = self._get_mime_for_format(format_type)
            return format_type, mime_type

        return DocumentFormat.UNKNOWN, "application/octet-stream"

    def _detect_from_content(self, file_path: Path) -> Tuple[DocumentFormat, str]:
        """Detect format from file content."""
        # Try filetype library first (fast, pure Python)
        kind = filetype.guess(str(file_path))
        if kind:
            mime_type = kind.mime
            format_type = self.MIME_MAPPING.get(mime_type, DocumentFormat.UNKNOWN)
            if format_type != DocumentFormat.UNKNOWN:
                return format_type, mime_type

        # Try python-magic if available
        if self._magic:
            try:
                mime_type = self._magic.from_file(str(file_path))
                format_type = self.MIME_MAPPING.get(mime_type, DocumentFormat.UNKNOWN)
                if format_type != DocumentFormat.UNKNOWN:
                    return format_type, mime_type
            except Exception:
                pass

        # Try to detect text files by reading first few bytes
        try:
            with open(file_path, "rb") as f:
                header = f.read(4096)
                if self._is_text(header):
                    # Try to detect specific text format
                    return self._detect_text_format(file_path, header)
        except Exception:
            pass

        return DocumentFormat.UNKNOWN, "application/octet-stream"

    def _is_text(self, data: bytes) -> bool:
        """Check if data appears to be text."""
        # Simple heuristic: check for null bytes or high ratio of non-printable chars
        if b"\x00" in data:
            return False

        # Check for valid UTF-8
        try:
            data.decode("utf-8")
            return True
        except UnicodeDecodeError:
            pass

        # Check for common text encodings
        for encoding in ["utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "latin-1"]:
            try:
                data.decode(encoding)
                return True
            except UnicodeDecodeError:
                continue

        return False

    def _detect_text_format(
        self, file_path: Path, header: bytes
    ) -> Tuple[DocumentFormat, str]:
        """Detect specific text format from content."""
        suffix = file_path.suffix.lower()

        # Try to decode as text
        text = ""
        for encoding in ["utf-8", "utf-8-sig", "latin-1"]:
            try:
                text = header.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        # Check for JSON
        if text.strip().startswith(("{", "[")):
            try:
                import json

                json.loads(text)
                return DocumentFormat.JSON, "application/json"
            except Exception:
                pass

        # Check for XML
        if text.strip().startswith("<"):
            if "<?xml" in text[:100].lower():
                return DocumentFormat.XML, "application/xml"
            if "<html" in text[:100].lower():
                return DocumentFormat.HTML, "text/html"

        # Check for YAML
        if suffix in (".yml", ".yaml") or (": " in text and "\n" in text):
            return DocumentFormat.YAML, "application/x-yaml"

        # Check for Markdown
        if suffix in (".md", ".markdown") or any(
            marker in text for marker in ["# ", "## ", "### ", "```"]
        ):
            return DocumentFormat.MARKDOWN, "text/markdown"

        # Check for CSV
        if "," in text and "\n" in text:
            lines = text.split("\n")[:3]
            if all("," in line for line in lines if line.strip()):
                return DocumentFormat.CSV, "text/csv"

        # Default to plain text
        return DocumentFormat.PLAIN_TEXT, "text/plain"

    def _get_mime_for_format(self, format_type: DocumentFormat) -> str:
        """Get MIME type for a format."""
        reverse_mapping = {v: k for k, v in self.MIME_MAPPING.items()}
        return reverse_mapping.get(format_type, "application/octet-stream")

    def is_supported(self, file_path: Path) -> bool:
        """Check if file format is supported."""
        format_type, _ = self.detect(file_path)
        return format_type != DocumentFormat.UNKNOWN

    def get_supported_extensions(self) -> list[str]:
        """Get list of supported file extensions."""
        extensions = [
            # Text
            ".txt",
            ".md",
            ".markdown",
            # Documents
            ".pdf",
            ".docx",
            ".xlsx",
            ".pptx",
            # Web
            ".html",
            ".htm",
            ".xml",
            # Data
            ".csv",
            ".json",
            ".yaml",
            ".yml",
            # Archives
            ".zip",
            ".tar",
            ".gz",
            # Code
            ".py",
            ".js",
            ".ts",
            ".java",
            ".cpp",
            ".c",
            ".go",
            ".rs",
            ".rb",
            ".php",
            # Jupyter
            ".ipynb",
        ]
        return extensions
