"""Document API endpoints for RLM Document Retrieval System."""

from pathlib import Path
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel

from rlm.config.settings import get_document_settings
from rlm.documents import Document, DocumentStorage, IngestionPipeline, ProcessingStatus
from rlm.documents.ingestion import BatchUploadResult
from rlm.documents.models import IngestionOptions

# Create router
router = APIRouter(prefix="/documents", tags=["documents"])

# Global instances (initialized on first use)
_storage: Optional[DocumentStorage] = None
_pipeline: Optional[IngestionPipeline] = None


def get_storage() -> DocumentStorage:
    """Get or create document storage instance."""
    global _storage
    if _storage is None:
        settings = get_document_settings()
        _storage = DocumentStorage(Path(settings.storage_path))
    return _storage


def get_pipeline() -> IngestionPipeline:
    """Get or create ingestion pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline(get_storage())
    return _pipeline


# Request/Response models
class DocumentResponse(BaseModel):
    """Document response model."""

    id: str
    filename: str
    file_size: int
    mime_type: str
    format_type: str
    status: str
    created_at: str
    updated_at: str
    error_message: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Document list response."""

    documents: List[DocumentResponse]
    total: int


class DocumentContentResponse(BaseModel):
    """Document content response."""

    document_id: str
    content: str
    word_count: int
    char_count: int


class BatchUploadResponse(BaseModel):
    """Batch upload response."""

    total_count: int
    success_count: int
    failed_count: int
    documents: List[DocumentResponse]
    errors: List[dict]


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> DocumentResponse:
    """
    Upload and process a document.

    Args:
        file: Document file to upload

    Returns:
        Document information
    """
    # Validate file
    settings = get_document_settings()
    if file.size and file.size > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB",
        )

    # Check file extension
    suffix = Path(file.filename).suffix.lower().lstrip(".")
    if suffix not in settings.allowed_formats:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File format '{suffix}' not supported",
        )

    try:
        # Save uploaded file temporarily
        temp_path = Path(f"/tmp/{file.filename}")
        content = await file.read()
        temp_path.write_bytes(content)

        # Process document
        pipeline = get_pipeline()
        document = await pipeline.ingest(temp_path, filename=file.filename)

        # Clean up temp file
        temp_path.unlink(missing_ok=True)

        return DocumentResponse(
            id=str(document.id),
            filename=document.filename,
            file_size=document.file_size,
            mime_type=document.mime_type,
            format_type=document.format_type.value,
            status=document.status.value,
            created_at=document.created_at.isoformat(),
            updated_at=document.updated_at.isoformat(),
            error_message=document.error_message,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}",
        )


@router.post("/upload/batch", response_model=BatchUploadResponse)
async def upload_batch(
    files: List[UploadFile] = File(...),
) -> BatchUploadResponse:
    """
    Upload multiple documents in batch.

    Args:
        files: List of document files to upload

    Returns:
        Batch upload results
    """
    settings = get_document_settings()
    pipeline = get_pipeline()

    # Save all files temporarily
    temp_paths = []
    for file in files:
        temp_path = Path(f"/tmp/{file.filename}")
        content = await file.read()
        temp_path.write_bytes(content)
        temp_paths.append(temp_path)

    try:
        # Process batch
        result = await pipeline.ingest_batch(temp_paths)

        # Clean up temp files
        for temp_path in temp_paths:
            temp_path.unlink(missing_ok=True)

        return BatchUploadResponse(
            total_count=result.total_count,
            success_count=result.success_count,
            failed_count=result.failed_count,
            documents=[
                DocumentResponse(
                    id=str(doc.id),
                    filename=doc.filename,
                    file_size=doc.file_size,
                    mime_type=doc.mime_type,
                    format_type=doc.format_type.value,
                    status=doc.status.value,
                    created_at=doc.created_at.isoformat(),
                    updated_at=doc.updated_at.isoformat(),
                    error_message=doc.error_message,
                )
                for doc in result.successful
            ],
            errors=result.failed,
        )

    except Exception as e:
        # Clean up temp files
        for temp_path in temp_paths:
            temp_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch upload failed: {str(e)}",
        )


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> DocumentListResponse:
    """
    List documents with optional filtering.

    Args:
        status: Filter by processing status
        limit: Maximum number of documents
        offset: Number of documents to skip

    Returns:
        List of documents
    """
    storage = get_storage()

    # Parse status filter
    status_filter = None
    if status:
        try:
            status_filter = ProcessingStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status}",
            )

    # Get documents
    documents = await storage.list_documents(
        status=status_filter, limit=limit, offset=offset
    )

    return DocumentListResponse(
        documents=[
            DocumentResponse(
                id=str(doc.id),
                filename=doc.filename,
                file_size=doc.file_size,
                mime_type=doc.mime_type,
                format_type=doc.format_type.value,
                status=doc.status.value,
                created_at=doc.created_at.isoformat(),
                updated_at=doc.updated_at.isoformat(),
                error_message=doc.error_message,
            )
            for doc in documents
        ],
        total=len(documents),
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str) -> DocumentResponse:
    """
    Get document by ID.

    Args:
        document_id: Document ID

    Returns:
        Document information
    """
    storage = get_storage()

    try:
        doc_id = UUID(document_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format",
        )

    document = await storage.get_document(doc_id)

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return DocumentResponse(
        id=str(document.id),
        filename=document.filename,
        file_size=document.file_size,
        mime_type=document.mime_type,
        format_type=document.format_type.value,
        status=document.status.value,
        created_at=document.created_at.isoformat(),
        updated_at=document.updated_at.isoformat(),
        error_message=document.error_message,
    )


@router.get("/{document_id}/content", response_model=DocumentContentResponse)
async def get_document_content(document_id: str) -> DocumentContentResponse:
    """
    Get document content by ID.

    Args:
        document_id: Document ID

    Returns:
        Document content
    """
    storage = get_storage()

    try:
        doc_id = UUID(document_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format",
        )

    # Get document metadata
    document = await storage.get_document(doc_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    # Get content
    content = await storage.get_document_content(doc_id)
    if content is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document content not found",
        )

    return DocumentContentResponse(
        document_id=document_id,
        content=content,
        word_count=len(content.split()),
        char_count=len(content),
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: str) -> None:
    """
    Delete a document.

    Args:
        document_id: Document ID
    """
    storage = get_storage()

    try:
        doc_id = UUID(document_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format",
        )

    deleted = await storage.delete_document(doc_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
