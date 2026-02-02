"""Document parsers package."""

from rlm.documents.parsers.base import DocumentParser
from rlm.documents.parsers.registry import (
    ParserRegistry,
    get_parser,
    get_registry,
    register_parser,
)

__all__ = [
    "DocumentParser",
    "ParserRegistry",
    "get_parser",
    "get_registry",
    "register_parser",
]
