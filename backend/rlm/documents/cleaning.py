"""Text cleaning and normalization utilities."""

import re
import unicodedata
from typing import List, Optional, Set

import ftfy


class TextCleaner:
    """Clean and normalize extracted text."""

    # Common PDF artifacts to remove
    PDF_ARTIFACTS = [
        r"\x0c",  # Form feed
        r"\ufeff",  # BOM
        r"\x00-\x08\x0b\x0c\x0e-\x1f",  # Control chars
    ]

    # Boilerplate patterns
    BOILERPLATE_PATTERNS = [
        r"Page \d+ of \d+",
        r"^\d+$",  # Page numbers
        r"Copyright ©? \d{4}",
        r"All rights reserved",
        r"Confidential",
        r"\[Date\]",
        r"\[Page \d+\]",
    ]

    # Whitespace patterns
    WHITESPACE_PATTERNS = [
        r" +",  # Multiple spaces
        r"\t+",  # Multiple tabs
        r"\n{3,}",  # More than 2 newlines
        r"\r\n",  # Windows line endings
        r"\r",  # Mac line endings
    ]

    def __init__(
        self,
        fix_encoding: bool = True,
        normalize_whitespace: bool = True,
        remove_artifacts: bool = True,
        remove_boilerplate: bool = False,
        deduplicate_lines: bool = True,
        remove_control_chars: bool = True,
    ) -> None:
        """
        Initialize text cleaner.

        Args:
            fix_encoding: Fix encoding issues
            normalize_whitespace: Normalize whitespace
            remove_artifacts: Remove PDF extraction artifacts
            remove_boilerplate: Remove common boilerplate text
            deduplicate_lines: Remove duplicate lines
            remove_control_chars: Remove control characters
        """
        self.fix_encoding = fix_encoding
        self.normalize_whitespace = normalize_whitespace
        self.remove_artifacts = remove_artifacts
        self.remove_boilerplate = remove_boilerplate
        self.deduplicate_lines = deduplicate_lines
        self.remove_control_chars = remove_control_chars

    def clean(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Fix encoding issues
        if self.fix_encoding:
            text = self._fix_encoding(text)

        # Remove control characters
        if self.remove_control_chars:
            text = self._remove_control_chars(text)

        # Remove PDF artifacts
        if self.remove_artifacts:
            text = self._remove_artifacts(text)

        # Normalize whitespace
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)

        # Remove boilerplate
        if self.remove_boilerplate:
            text = self._remove_boilerplate(text)

        # Deduplicate lines
        if self.deduplicate_lines:
            text = self._deduplicate_lines(text)

        # Final cleanup
        text = self._final_cleanup(text)

        return text

    def _fix_encoding(self, text: str) -> str:
        """Fix encoding issues using ftfy."""
        # Fix mojibake and encoding issues
        text = ftfy.fix_text(text)

        # Normalize unicode
        text = unicodedata.normalize("NFC", text)

        return text

    def _remove_control_chars(self, text: str) -> str:
        """Remove control characters except newlines and tabs."""
        # Keep newlines (\n) and tabs (\t), remove other control chars
        cleaned = []
        for char in text:
            code = ord(char)
            # Keep printable chars, newlines, and tabs
            if code >= 32 or code in (9, 10):  # 9=tab, 10=newline
                cleaned.append(char)
            elif code == 13:  # Carriage return -> newline
                cleaned.append("\n")

        return "".join(cleaned)

    def _remove_artifacts(self, text: str) -> str:
        """Remove PDF extraction artifacts."""
        for artifact in self.PDF_ARTIFACTS:
            text = re.sub(artifact, "", text)

        # Remove common PDF extraction noise
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)  # Excessive newlines
        text = re.sub(r"[ ]+", " ", text)  # Multiple spaces
        text = re.sub(r"\t[ ]*", " ", text)  # Tabs followed by spaces

        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace."""
        # Convert Windows/Mac line endings to Unix
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Normalize tabs and spaces
        text = re.sub(r"[ \t]+", " ", text)

        # Remove trailing whitespace
        lines = [line.rstrip() for line in text.split("\n")]

        # Remove empty lines at start and end
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

        # Normalize multiple blank lines
        result = []
        prev_empty = False
        for line in lines:
            is_empty = not line.strip()
            if is_empty and prev_empty:
                continue  # Skip consecutive empty lines
            result.append(line)
            prev_empty = is_empty

        return "\n".join(result)

    def _remove_boilerplate(self, text: str) -> str:
        """Remove common boilerplate text."""
        lines = text.split("\n")
        filtered = []

        for line in lines:
            keep = True
            for pattern in self.BOILERPLATE_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    keep = False
                    break
            if keep:
                filtered.append(line)

        return "\n".join(filtered)

    def _deduplicate_lines(self, text: str) -> str:
        """Remove duplicate consecutive lines."""
        lines = text.split("\n")
        result = []
        seen: Set[str] = set()

        for line in lines:
            normalized = line.strip().lower()
            if normalized and normalized in seen:
                continue
            result.append(line)
            if normalized:
                seen.add(normalized)
                # Limit set size to prevent memory issues
                if len(seen) > 1000:
                    seen.clear()

        return "\n".join(result)

    def _final_cleanup(self, text: str) -> str:
        """Final cleanup pass."""
        # Remove leading/trailing whitespace
        text = text.strip()

        # Ensure text ends with single newline
        text = text.rstrip() + "\n"

        return text


def clean_text(
    text: str,
    fix_encoding: bool = True,
    normalize_whitespace: bool = True,
    remove_artifacts: bool = True,
    remove_boilerplate: bool = False,
    deduplicate_lines: bool = True,
) -> str:
    """
    Convenience function to clean text with default options.

    Args:
        text: Raw text to clean
        fix_encoding: Fix encoding issues
        normalize_whitespace: Normalize whitespace
        remove_artifacts: Remove PDF artifacts
        remove_boilerplate: Remove boilerplate
        deduplicate_lines: Deduplicate lines

    Returns:
        Cleaned text
    """
    cleaner = TextCleaner(
        fix_encoding=fix_encoding,
        normalize_whitespace=normalize_whitespace,
        remove_artifacts=remove_artifacts,
        remove_boilerplate=remove_boilerplate,
        deduplicate_lines=deduplicate_lines,
    )
    return cleaner.clean(text)
