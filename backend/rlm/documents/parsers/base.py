"""Parser base interface for document ingestion."""

from abc import ABC, abstractmethod
from pathlib import Path

from rlm.documents.models import DocumentContent, DocumentFormat, ParseOptions


class DocumentParser(ABC):
    """Base interface for document parsers."""

    @abstractmethod
    async def parse(
        self, file_path: Path, options: ParseOptions
    ) -> DocumentContent:
        """
        Parse a document and extract its content.

        Args:
            file_path: Path to the document file
            options: Parsing options

        Returns:
            DocumentContent with extracted text and metadata

        Raises:
            ParseError: If parsing fails
        """
        pass

    @abstractmethod
    def supports(self, format_type: DocumentFormat) -> bool:
        """
        Check if this parser supports the given format.

        Args:
            format_type: Document format to check

        Returns:
            True if the format is supported
        """
        pass

    @property
    @abstractmethod
    def priority(self) -> int:
        """
        Parser priority (higher = preferred).

        Returns:
            Priority value for this parser
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Parser name.

        Returns:
            Human-readable parser name
        """
        pass

    def can_handle(self, file_path: Path, format_type: DocumentFormat) -> bool:
        """
        Check if this parser can handle the given file.

        Args:
            file_path: Path to the file
            format_type: Detected format type

        Returns:
            True if this parser can handle the file
        """
        return self.supports(format_type)
