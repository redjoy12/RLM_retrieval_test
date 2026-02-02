"""Parser registry for managing document parsers."""

from pathlib import Path
from typing import Dict, List, Optional, Type

from rlm.documents.models import DocumentFormat
from rlm.documents.parsers.base import DocumentParser


class ParserRegistry:
    """Registry for document parsers."""

    def __init__(self) -> None:
        """Initialize the parser registry."""
        self._parsers: Dict[DocumentFormat, List[DocumentParser]] = {}
        self._all_parsers: List[DocumentParser] = []

    def register(self, parser: DocumentParser) -> None:
        """
        Register a parser.

        Args:
            parser: Parser instance to register
        """
        # Register for all supported formats
        for fmt in DocumentFormat:
            if parser.supports(fmt) and fmt != DocumentFormat.UNKNOWN:
                if fmt not in self._parsers:
                    self._parsers[fmt] = []
                self._parsers[fmt].append(parser)
                # Sort by priority (descending)
                self._parsers[fmt].sort(key=lambda p: p.priority, reverse=True)

        if parser not in self._all_parsers:
            self._all_parsers.append(parser)

    def unregister(self, parser: DocumentParser) -> None:
        """
        Unregister a parser.

        Args:
            parser: Parser instance to unregister
        """
        for fmt in self._parsers:
            if parser in self._parsers[fmt]:
                self._parsers[fmt].remove(parser)

        if parser in self._all_parsers:
            self._all_parsers.remove(parser)

    def get_parser(
        self,
        format_type: DocumentFormat,
        preferred_name: Optional[str] = None,
    ) -> Optional[DocumentParser]:
        """
        Get the best parser for a format.

        Args:
            format_type: Document format
            preferred_name: Optional preferred parser name

        Returns:
            Best matching parser or None
        """
        parsers = self._parsers.get(format_type, [])

        if not parsers:
            return None

        if preferred_name:
            for parser in parsers:
                if parser.name == preferred_name:
                    return parser

        # Return highest priority parser
        return parsers[0]

    def get_all_parsers(self, format_type: DocumentFormat) -> List[DocumentParser]:
        """
        Get all parsers for a format.

        Args:
            format_type: Document format

        Returns:
            List of parsers (sorted by priority)
        """
        return self._parsers.get(format_type, [])

    def list_parsers(self) -> List[DocumentParser]:
        """List all registered parsers."""
        return self._all_parsers.copy()

    def supports(self, format_type: DocumentFormat) -> bool:
        """Check if any parser supports the format."""
        return format_type in self._parsers and len(self._parsers[format_type]) > 0


# Global registry instance
_registry: Optional[ParserRegistry] = None


def get_registry() -> ParserRegistry:
    """Get the global parser registry."""
    global _registry
    if _registry is None:
        _registry = ParserRegistry()
    return _registry


def register_parser(parser: DocumentParser) -> None:
    """Register a parser globally."""
    get_registry().register(parser)


def get_parser(
    format_type: DocumentFormat, preferred_name: Optional[str] = None
) -> Optional[DocumentParser]:
    """Get parser for a format."""
    return get_registry().get_parser(format_type, preferred_name)
