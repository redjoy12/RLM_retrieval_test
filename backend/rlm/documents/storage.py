"""Document storage manager for local filesystem storage."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from uuid import UUID

import aiofiles

from rlm.documents.models import Document, ProcessingStatus


class DocumentStorage:
    """Local filesystem storage for documents."""

    def __init__(self, storage_path: Path) -> None:
        """
        Initialize document storage.

        Args:
            storage_path: Base path for document storage
        """
        self.storage_path = Path(storage_path)
        self.documents_dir = self.storage_path / "documents"
        self.content_dir = self.storage_path / "content"
        self.metadata_dir = self.storage_path / "metadata"

        # Create directories
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure storage directories exist."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(exist_ok=True)
        self.content_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)

    async def save_file(
        self, file_path: Path, document_id: UUID, filename: str
    ) -> Path:
        """
        Save uploaded file to storage.

        Args:
            file_path: Path to uploaded file
            document_id: Document ID
            filename: Original filename

        Returns:
            Path to stored file
        """
        # Create document directory
        doc_dir = self.documents_dir / str(document_id)
        doc_dir.mkdir(exist_ok=True)

        # Determine safe filename
        safe_name = self._sanitize_filename(filename)
        dest_path = doc_dir / safe_name

        # Copy file
        await self._async_copy(file_path, dest_path)

        return dest_path

    async def save_document(self, document: Document) -> None:
        """
        Save document metadata.

        Args:
            document: Document to save
        """
        # Save metadata
        metadata_path = self.metadata_dir / f"{document.id}.json"
        async with aiofiles.open(metadata_path, "w", encoding="utf-8") as f:
            await f.write(document.model_dump_json(indent=2))

        # Save content
        if document.content.cleaned_text:
            content_path = self.content_dir / f"{document.id}.txt"
            async with aiofiles.open(content_path, "w", encoding="utf-8") as f:
                await f.write(document.content.cleaned_text)

            # Save chunks metadata
            chunks_meta = {
                "document_id": str(document.id),
                "total_chunks": len(document.content.chunks),
                "total_tokens": document.content.total_tokens,
                "chunks": [
                    {
                        "index": chunk.index,
                        "token_count": chunk.token_count,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                    }
                    for chunk in document.content.chunks
                ],
            }
            chunks_path = self.content_dir / f"{document.id}_chunks.json"
            async with aiofiles.open(chunks_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(chunks_meta, indent=2))

    async def get_document(self, document_id: UUID) -> Optional[Document]:
        """
        Get document by ID.

        Args:
            document_id: Document ID

        Returns:
            Document or None if not found
        """
        metadata_path = self.metadata_dir / f"{document_id}.json"

        if not metadata_path.exists():
            return None

        try:
            async with aiofiles.open(metadata_path, "r", encoding="utf-8") as f:
                data = await f.read()
                return Document.model_validate_json(data)
        except Exception:
            return None

    async def get_document_content(self, document_id: UUID) -> Optional[str]:
        """
        Get document content by ID.

        Args:
            document_id: Document ID

        Returns:
            Content or None if not found
        """
        content_path = self.content_dir / f"{document_id}.txt"

        if not content_path.exists():
            return None

        try:
            async with aiofiles.open(content_path, "r", encoding="utf-8") as f:
                return await f.read()
        except Exception:
            return None

    async def list_documents(
        self, status: Optional[ProcessingStatus] = None, limit: int = 100, offset: int = 0
    ) -> List[Document]:
        """
        List documents with optional filtering.

        Args:
            status: Filter by status
            limit: Maximum number of documents
            offset: Skip first N documents

        Returns:
            List of documents
        """
        documents = []

        # Get all metadata files
        metadata_files = sorted(self.metadata_dir.glob("*.json"))

        # Apply offset
        metadata_files = metadata_files[offset:offset + limit * 2]  # Fetch more for filtering

        for metadata_file in metadata_files:
            try:
                async with aiofiles.open(metadata_file, "r", encoding="utf-8") as f:
                    data = await f.read()
                    doc = Document.model_validate_json(data)

                    if status is None or doc.status == status:
                        documents.append(doc)

                        if len(documents) >= limit:
                            break
            except Exception:
                continue

        return documents[:limit]

    async def delete_document(self, document_id: UUID) -> bool:
        """
        Delete document and associated files.

        Args:
            document_id: Document ID

        Returns:
            True if deleted, False if not found
        """
        try:
            # Delete metadata
            metadata_path = self.metadata_dir / f"{document_id}.json"
            if metadata_path.exists():
                metadata_path.unlink()

            # Delete content
            content_path = self.content_dir / f"{document_id}.txt"
            if content_path.exists():
                content_path.unlink()

            chunks_path = self.content_dir / f"{document_id}_chunks.json"
            if chunks_path.exists():
                chunks_path.unlink()

            # Delete document directory
            doc_dir = self.documents_dir / str(document_id)
            if doc_dir.exists():
                shutil.rmtree(doc_dir)

            return True
        except Exception:
            return False

    async def update_status(
        self, document_id: UUID, status: ProcessingStatus, error_message: Optional[str] = None
    ) -> bool:
        """
        Update document processing status.

        Args:
            document_id: Document ID
            status: New status
            error_message: Optional error message

        Returns:
            True if updated, False if not found
        """
        doc = await self.get_document(document_id)
        if not doc:
            return False

        doc.status = status
        doc.updated_at = datetime.utcnow()
        if error_message:
            doc.error_message = error_message

        await self.save_document(doc)
        return True

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage."""
        # Remove potentially dangerous characters
        safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        sanitized = "".join(c if c in safe_chars else "_" for c in filename)
        return sanitized[:255]  # Limit length

    async def _async_copy(self, src: Path, dst: Path) -> None:
        """Asynchronously copy file."""
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, shutil.copy2, src, dst)
