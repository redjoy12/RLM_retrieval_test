"""Configuration settings for RLM Document Retrieval System."""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RLMSettings(BaseSettings):
    """RLM Core Engine settings."""
    
    model_config = SettingsConfigDict(
        env_prefix="RLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Recursion settings
    max_recursion_depth: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum recursion depth for sub-LLM calls",
    )
    max_sub_llm_calls: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum total sub-LLM calls per query",
    )
    
    # Timeout settings (seconds)
    code_execution_timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Timeout for code execution in REPL",
    )
    llm_timeout: int = Field(
        default=60,
        ge=1,
        le=600,
        description="Timeout for LLM API calls",
    )
    
    # Sandbox settings
    sandbox_output_limit: int = Field(
        default=8192,
        ge=1024,
        le=65536,
        description="Maximum output characters from REPL per execution",
    )
    sandbox_memory_limit_mb: int = Field(
        default=512,
        ge=128,
        le=4096,
        description="Memory limit for REPL execution in MB",
    )
    
    # Context settings for 10M+ token support
    context_chunk_size: int = Field(
        default=100000,  # ~25K tokens per chunk
        ge=10000,
        le=500000,
        description="Chunk size in characters for large documents",
    )
    max_context_chunks_in_memory: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum chunks to keep in memory at once",
    )
    
    # LLM settings
    default_model: str = Field(
        default="gpt-5-mini",
        description="Default LLM model (supports any LiteLLM-compatible model)",
    )
    default_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Default temperature for LLM calls",
    )
    default_max_tokens: Optional[int] = Field(
        default=None,
        description="Default max tokens for LLM responses",
    )
    
    # LiteLLM settings
    litellm_provider: str = Field(
        default="openai",
        description="LiteLLM provider (openai, anthropic, azure, etc.)",
    )
    litellm_api_base: Optional[str] = Field(
        default=None,
        description="Custom API base URL for LiteLLM",
    )
    litellm_retry_count: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of retries for failed LLM calls",
    )
    
    # Logging settings
    log_level: str = Field(
        default="INFO",
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
        description="Logging level",
    )
    log_dir: str = Field(
        default="./logs",
        description="Directory for trajectory logs",
    )
    enable_trajectory_logging: bool = Field(
        default=True,
        description="Enable trajectory logging",
    )
    
    # API settings
    api_host: str = Field(
        default="0.0.0.0",
        description="API server host",
    )
    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="API server port",
    )
    
    @field_validator("default_model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate that the model string is not empty."""
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Normalize log level."""
        return v.upper()


class SandboxSettings(BaseSettings):
    """Sandbox-specific settings."""
    
    model_config = SettingsConfigDict(
        env_prefix="SANDBOX_",
        env_file=".env",
        extra="ignore",
    )
    
    # Allowed modules in REPL
    allowed_modules: List[str] = Field(
        default=[
            "json",
            "re",
            "math",
            "random",
            "datetime",
            "collections",
            "itertools",
            "statistics",
            "typing",
            "string",
            "hashlib",
            "base64",
            "binascii",
            "decimal",
            "fractions",
            "numbers",
        ],
        description="List of allowed Python modules in REPL",
    )
    
    # Blocked builtins
    blocked_builtins: List[str] = Field(
        default=[
            "__import__",
            "eval",
            "compile",
            "exec",
            "open",
            "exit",
            "quit",
        ],
        description="List of blocked builtin functions",
    )
    
    # Code limits
    max_code_length: int = Field(
        default=100000,
        ge=1000,
        le=1000000,
        description="Maximum code length in characters",
    )
    max_loops: int = Field(
        default=10000,
        ge=100,
        le=1000000,
        description="Maximum loop iterations",
    )


@lru_cache()
def get_settings() -> RLMSettings:
    """Get cached settings instance.
    
    Returns:
        RLMSettings instance
    """
    return RLMSettings()


class DockerSandboxSettings(BaseSettings):
    """Docker sandbox-specific settings."""
    
    model_config = SettingsConfigDict(
        env_prefix="DOCKER_",
        env_file=".env",
        extra="ignore",
    )
    
    # Docker image settings
    image: str = Field(
        default="python:3.11-slim",
        description="Docker image for sandbox containers",
    )
    
    # Resource limits
    memory_limit: str = Field(
        default="512m",
        description="Memory limit for containers (e.g., 512m, 1g)",
    )
    cpu_limit: float = Field(
        default=1.0,
        ge=0.1,
        le=16.0,
        description="CPU limit in cores",
    )
    
    # Security settings
    security_profile: str = Field(
        default="standard",
        pattern="^(strict|standard|development)$",
        description="Security profile: strict, standard, or development",
    )
    network_enabled: bool = Field(
        default=False,
        description="Enable network access in containers",
    )
    
    # Container management
    auto_cleanup: bool = Field(
        default=True,
        description="Automatically remove containers after execution",
    )
    container_timeout: int = Field(
        default=60,
        ge=10,
        le=600,
        description="Container startup timeout in seconds",
    )
    
    # Volume settings
    volume_mounts: List[str] = Field(
        default_factory=list,
        description="Additional volume mounts (format: host:container)",
    )
    
    @field_validator("memory_limit")
    @classmethod
    def validate_memory_limit(cls, v: str) -> str:
        """Validate memory limit format."""
        v = v.lower().strip()
        valid_suffixes = ["k", "m", "g", "t", "b"]
        
        # Remove number and check if suffix is valid
        suffix = "".join(c for c in v if c.isalpha())
        number_part = "".join(c for c in v if c.isdigit())
        
        if not number_part:
            raise ValueError(f"Memory limit must include a number: {v}")
        
        if suffix and suffix not in valid_suffixes:
            raise ValueError(
                f"Invalid memory suffix: {suffix}. "
                f"Valid: {valid_suffixes}"
            )
        
        return v


@lru_cache()
def get_sandbox_settings() -> SandboxSettings:
    """Get cached sandbox settings instance.
    
    Returns:
        SandboxSettings instance
    """
    return SandboxSettings()


@lru_cache()
def get_docker_settings() -> DockerSandboxSettings:
    """Get cached Docker sandbox settings instance.
    
    Returns:
        DockerSandboxSettings instance
    """
    return DockerSandboxSettings()


class LLMClientSettings(BaseSettings):
    """Component 3: Multi-Model LLM Client settings."""
    
    model_config = SettingsConfigDict(
        env_prefix="LLMCLIENT_",
        env_file=".env",
        extra="ignore",
    )
    
    # Feature toggles
    enable_rate_limiting: bool = Field(
        default=True,
        description="Enable rate limiting for LLM calls",
    )
    enable_cost_tracking: bool = Field(
        default=True,
        description="Enable cost tracking per query",
    )
    enable_circuit_breaker: bool = Field(
        default=True,
        description="Enable circuit breaker for failover",
    )
    enable_caching: bool = Field(
        default=True,
        description="Enable response caching",
    )
    
    # Rate limiting settings
    openai_rpm: int = Field(
        default=60,
        ge=1,
        description="OpenAI requests per minute",
    )
    openai_tpm: int = Field(
        default=60000,
        ge=1,
        description="OpenAI tokens per minute",
    )
    anthropic_rpm: int = Field(
        default=50,
        ge=1,
        description="Anthropic requests per minute",
    )
    anthropic_tpm: int = Field(
        default=40000,
        ge=1,
        description="Anthropic tokens per minute",
    )
    local_rpm: int = Field(
        default=1000,
        ge=1,
        description="Local LLM requests per minute",
    )
    
    # Cost tracking settings
    cost_log_file: Optional[str] = Field(
        default="./logs/costs.jsonl",
        description="Path to cost log file",
    )
    
    # Circuit breaker settings
    circuit_failure_threshold: int = Field(
        default=5,
        ge=1,
        description="Failures before opening circuit",
    )
    circuit_recovery_timeout: int = Field(
        default=60,
        ge=1,
        description="Seconds before retry in half-open state",
    )
    
    # Cache settings
    cache_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        description="Cache time-to-live in seconds",
    )
    cache_max_size: int = Field(
        default=1000,
        ge=100,
        description="Maximum cache entries",
    )
    
    # Batch processing settings
    batch_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Default batch size for parallel requests",
    )
    batch_timeout: int = Field(
        default=30,
        ge=1,
        description="Batch execution timeout in seconds",
    )
    max_concurrent_requests: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum concurrent LLM requests",
    )


@lru_cache()
def get_llm_client_settings() -> LLMClientSettings:
    """Get cached LLM client settings instance.
    
    Returns:
        LLMClientSettings instance
    """
    return LLMClientSettings()


class DocumentSettings(BaseSettings):
    """Component 4: Document Ingestion Pipeline settings."""
    
    model_config = SettingsConfigDict(
        env_prefix="DOCUMENT_",
        env_file=".env",
        extra="ignore",
    )
    
    # Storage settings
    storage_path: str = Field(
        default="./data/documents",
        description="Path for document storage",
    )
    max_file_size_mb: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum file size in MB",
    )
    
    # Processing settings
    default_chunk_size: int = Field(
        default=100000,  # ~25K tokens
        ge=10000,
        le=500000,
        description="Default chunk size in characters",
    )
    chunk_overlap: int = Field(
        default=1000,
        ge=0,
        le=10000,
        description="Overlap between chunks in characters",
    )
    max_chunks_per_document: int = Field(
        default=1000,
        ge=10,
        le=10000,
        description="Maximum chunks per document",
    )
    
    # Parser settings
    pdf_parser: str = Field(
        default="auto",
        pattern="^(auto|pymupdf|docling|unstructured)$",
        description="PDF parser: auto, pymupdf, docling, or unstructured",
    )
    enable_unstructured: bool = Field(
        default=True,
        description="Enable unstructured.io parser",
    )
    
    # Feature toggles
    enable_cleaning: bool = Field(
        default=True,
        description="Enable text cleaning",
    )
    enable_chunking: bool = Field(
        default=True,
        description="Enable content chunking",
    )
    enable_metadata_extraction: bool = Field(
        default=True,
        description="Enable metadata extraction",
    )
    use_llm_metadata: bool = Field(
        default=False,
        description="Use LLM for advanced metadata extraction",
    )
    
    # Performance settings
    max_concurrent_uploads: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum concurrent uploads",
    )
    upload_timeout_seconds: int = Field(
        default=300,
        ge=30,
        le=1800,
        description="Upload timeout in seconds",
    )
    parse_timeout_seconds: int = Field(
        default=300,
        ge=30,
        le=1800,
        description="Document parsing timeout in seconds",
    )
    
    # Supported formats (whitelist)
    allowed_formats: List[str] = Field(
        default=[
            "txt", "md", "pdf", "docx", "xlsx", "pptx",
            "html", "csv", "json", "xml", "zip", "py",
        ],
        description="List of allowed file formats",
    )


@lru_cache()
def get_document_settings() -> DocumentSettings:
    """Get cached document settings instance.
    
    Returns:
        DocumentSettings instance
    """
    return DocumentSettings()


class SubLLMManagerSettings(BaseSettings):
    """Component 5: Async Sub-LLM Manager settings."""
    
    model_config = SettingsConfigDict(
        env_prefix="SUBLLM_",
        env_file=".env",
        extra="ignore",
    )
    
    # Concurrency settings
    max_concurrent: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum concurrent sub-LLM calls",
    )
    connection_limit: int = Field(
        default=100,
        ge=10,
        le=500,
        description="HTTP connection pool limit",
    )
    keepalive_connections: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Keep-alive connections in pool",
    )
    
    # Caching settings
    enable_caching: bool = Field(
        default=True,
        description="Enable session-level response caching",
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Cache time-to-live in seconds",
    )
    cache_max_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Maximum cache entries per session",
    )
    
    # Batching settings
    enable_batching: bool = Field(
        default=True,
        description="Enable query batching for efficiency",
    )
    batch_size: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum calls per batch",
    )
    batch_timeout_ms: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Batch collection timeout in milliseconds",
    )
    
    # Resilience settings
    retry_attempts: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of retry attempts per call",
    )
    retry_backoff: float = Field(
        default=1.5,
        ge=1.0,
        le=5.0,
        description="Exponential backoff multiplier",
    )
    timeout_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
        description="Timeout per sub-LLM call",
    )
    
    # Streaming settings
    streaming_enabled: bool = Field(
        default=True,
        description="Enable streaming for sub-LLM calls",
    )
    stream_chunk_timeout: float = Field(
        default=30.0,
        ge=5.0,
        le=120.0,
        description="Timeout between stream chunks",
    )
    stream_buffer_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Stream buffer size per call",
    )
    
    # Connection pool settings
    enable_connection_pool: bool = Field(
        default=True,
        description="Enable HTTP connection pooling",
    )
    connection_timeout: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
        description="Connection timeout in seconds",
    )


@lru_cache()
def get_sub_llm_settings() -> SubLLMManagerSettings:
    """Get cached Sub-LLM Manager settings instance.
    
    Returns:
        SubLLMManagerSettings instance
    """
    return SubLLMManagerSettings()


class RoutingSettings(BaseSettings):
    """Component 7: Query Router & Optimizer settings."""
    
    model_config = SettingsConfigDict(
        env_prefix="ROUTING_",
        env_file=".env",
        extra="ignore",
    )
    
    # Strategy thresholds
    direct_llm_max_context: int = Field(
        default=10000,
        ge=1000,
        le=100000,
        description="Max context size (chars) for direct LLM strategy",
    )
    rag_max_context: int = Field(
        default=500000,
        ge=50000,
        le=10000000,
        description="Max context size (chars) for RAG strategy",
    )
    
    # Cost estimation
    enable_tiktoken: bool = Field(
        default=True,
        description="Use tiktoken for accurate token counting",
    )
    cost_buffer_percent: int = Field(
        default=10,
        ge=0,
        le=50,
        description="Safety buffer percentage for cost estimates",
    )
    
    # Qdrant settings for RAG
    qdrant_host: str = Field(
        default="localhost",
        description="Qdrant server host",
    )
    qdrant_port: int = Field(
        default=6333,
        ge=1,
        le=65535,
        description="Qdrant server port",
    )
    qdrant_collection: str = Field(
        default="rlm_chunks",
        description="Qdrant collection name",
    )
    qdrant_https: bool = Field(
        default=False,
        description="Use HTTPS for Qdrant connection",
    )
    qdrant_api_key: Optional[str] = Field(
        default=None,
        description="Qdrant API key (if required)",
    )
    
    # Embedding settings
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model",
    )
    embedding_dimensions: int = Field(
        default=1536,
        ge=128,
        le=4096,
        description="Embedding vector dimensions",
    )
    embedding_batch_size: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Batch size for embedding generation",
    )
    
    # RAG settings
    rag_default_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Default number of chunks to retrieve",
    )
    rag_score_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for RAG results",
    )
    
    # Hybrid strategy settings
    hybrid_rag_top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of chunks for hybrid RAG phase",
    )
    hybrid_filter_ratio: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description="Ratio of chunks to pass to RLM in hybrid mode",
    )
    
    # Query optimization
    enable_query_optimization: bool = Field(
        default=True,
        description="Enable LLM-based query optimization",
    )
    optimizer_model: str = Field(
        default="gpt-5-mini",
        description="Model for query optimization",
    )
    optimization_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold to use optimized query",
    )
    
    # Routing visibility
    show_routing_info: bool = Field(
        default=True,
        description="Include routing decision in API responses",
    )


@lru_cache()
def get_routing_settings() -> RoutingSettings:
    """Get cached Routing settings instance.
    
    Returns:
        RoutingSettings instance
    """
    return RoutingSettings()


class HybridSettings(BaseSettings):
    """Component 8: Hybrid RAG Integration settings."""
    
    model_config = SettingsConfigDict(
        env_prefix="HYBRID_",
        env_file=".env",
        extra="ignore",
    )
    
    # Hybrid search weights
    semantic_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for semantic (vector) search in hybrid fusion",
    )
    keyword_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for keyword (BM25) search in hybrid fusion",
    )
    
    # RRF (Reciprocal Rank Fusion) settings
    rrf_k: int = Field(
        default=60,
        ge=1,
        le=100,
        description="RRF fusion constant (higher = more bias to top ranks)",
    )
    
    # Reranking settings
    enable_reranking: bool = Field(
        default=True,
        description="Enable cross-encoder reranking",
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking",
    )
    reranker_top_k: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Number of chunks to rerank",
    )
    reranker_final_top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of chunks after reranking",
    )
    reranker_device: str = Field(
        default="cpu",
        pattern="^(cpu|cuda|auto)$",
        description="Device for reranker (cpu, cuda, auto)",
    )
    reranker_batch_size: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Batch size for reranker inference",
    )
    
    # LLM reranker options (for multi-stage)
    enable_llm_reranker: bool = Field(
        default=False,
        description="Enable LLM-based reranking (slower, more accurate)",
    )
    llm_reranker_model: str = Field(
        default="gpt-5-mini",
        description="LLM model for reranking",
    )
    llm_reranker_max_chunks: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Max chunks for LLM reranker",
    )
    
    # Adaptive selection settings
    enable_adaptive_selection: bool = Field(
        default=True,
        description="Enable adaptive chunk selection",
    )
    adaptive_min_chunks: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Minimum chunks for adaptive selection",
    )
    adaptive_max_chunks: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Maximum chunks for adaptive selection",
    )
    adaptive_diversity_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Jaccard similarity threshold for diversity",
    )
    
    # Citation settings
    enable_citations: bool = Field(
        default=True,
        description="Enable citation tracking in answers",
    )
    citation_format: str = Field(
        default="numbered",
        pattern="^(numbered|document_grouped)$",
        description="Citation format style",
    )
    citation_max_preview_length: int = Field(
        default=100,
        ge=50,
        le=500,
        description="Max preview length for citations",
    )
    
    # BM25 settings
    bm25_k1: float = Field(
        default=1.5,
        ge=0.5,
        le=3.0,
        description="BM25 term frequency saturation parameter",
    )
    bm25_b: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="BM25 length normalization parameter",
    )
    
    # Performance settings
    hybrid_search_timeout_ms: int = Field(
        default=5000,
        ge=1000,
        le=30000,
        description="Timeout for hybrid search in milliseconds",
    )
    enable_hybrid_cache: bool = Field(
        default=True,
        description="Enable caching for hybrid search results",
    )
    hybrid_cache_ttl_seconds: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Hybrid search cache TTL",
    )


@lru_cache()
def get_hybrid_settings() -> HybridSettings:
    """Get cached Hybrid settings instance.
    
    Returns:
        HybridSettings instance
    """
    return HybridSettings()


class SessionSettings(BaseSettings):
    """Session Management settings - Component 9."""
    
    model_config = SettingsConfigDict(
        env_prefix="SESSION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Database settings
    db_path: str = Field(
        default="./data/sessions.db",
        description="Path to SQLite database for sessions",
    )
    
    # Context window settings
    max_tokens_per_session: int = Field(
        default=128000,
        ge=1000,
        le=200000,
        description="Maximum tokens per session context window",
    )
    
    # Compaction settings
    compaction_threshold: float = Field(
        default=0.8,
        ge=0.5,
        le=0.95,
        description="Threshold (0-1) to trigger context compaction",
    )
    retention_window: int = Field(
        default=6,
        ge=2,
        le=20,
        description="Number of recent messages to always retain",
    )
    enable_auto_compaction: bool = Field(
        default=True,
        description="Automatically compact when threshold reached",
    )
    compaction_model: str = Field(
        default="gpt-5-nano",
        description="Model to use for generating compaction summaries",
    )
    
    # Session lifecycle settings
    session_ttl_hours: int = Field(
        default=72,  # 3 days
        ge=1,
        le=720,  # 30 days max
        description="Session time-to-live in hours",
    )
    max_sessions_per_user: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum active sessions per user",
    )
    
    # FTS5 settings
    enable_fts_search: bool = Field(
        default=True,
        description="Enable FTS5 for conversation search",
    )
    fts_max_results: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Maximum FTS search results",
    )
    
    # Component 8 integration defaults
    default_semantic_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Default semantic weight for hybrid search",
    )
    default_keyword_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Default keyword weight for hybrid search",
    )
    default_enable_reranking: bool = Field(
        default=True,
        description="Default reranking setting for new sessions",
    )
    default_enable_citations: bool = Field(
        default=True,
        description="Default citation setting for new sessions",
    )
    context_enhancement_history_size: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of previous queries to use for search enhancement",
    )
    
    # Cleanup settings
    enable_auto_cleanup: bool = Field(
        default=True,
        description="Enable automatic cleanup of expired sessions",
    )
    cleanup_interval_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Interval between cleanup runs in hours",
    )


@lru_cache()
def get_session_settings() -> SessionSettings:
    """Get cached Session settings instance.
    
    Returns:
        SessionSettings instance
    """
    return SessionSettings()
