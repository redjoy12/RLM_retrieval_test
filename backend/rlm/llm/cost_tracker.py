"""Cost tracking for LLM API calls."""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from rlm.llm.model_registry import get_model_registry
from rlm.types import LLMResponse

logger = structlog.get_logger()


@dataclass
class CostEntry:
    """A single cost entry for an LLM call."""

    timestamp: datetime
    session_id: str
    provider: str
    model: str
    query_type: str  # "root" or "sub_llm"
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "provider": self.provider,
            "model": self.model,
            "query_type": self.query_type,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CostEntry":
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            session_id=data["session_id"],
            provider=data["provider"],
            model=data["model"],
            query_type=data["query_type"],
            prompt_tokens=data["prompt_tokens"],
            completion_tokens=data["completion_tokens"],
            total_tokens=data["total_tokens"],
            cost_usd=data["cost_usd"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class CostReport:
    """Cost report for a session or time period."""

    total_cost: float
    total_tokens: int
    total_requests: int
    by_provider: Dict[str, float]
    by_model: Dict[str, float]
    by_query_type: Dict[str, float]
    entries: List[CostEntry]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "total_requests": self.total_requests,
            "by_provider": self.by_provider,
            "by_model": self.by_model,
            "by_query_type": self.by_query_type,
            "entries": [e.to_dict() for e in self.entries],
        }


class CostTracker:
    """Track costs for LLM API calls."""

    def __init__(
        self,
        log_file: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        """Initialize cost tracker.

        Args:
            log_file: Path to JSONL log file for cost entries
            enabled: Whether cost tracking is enabled
        """
        self.enabled = enabled
        self._entries: List[CostEntry] = []
        self._registry = get_model_registry()

        if log_file:
            self._log_file = Path(log_file)
            # Ensure directory exists
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            self._log_file = None

        logger.info(
            "cost_tracker_initialized",
            enabled=enabled,
            log_file=str(log_file) if log_file else None,
        )

    def calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Calculate cost for a request using model registry pricing.

        Args:
            model: Model identifier
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        return self._registry.calculate_cost(
            model, prompt_tokens, completion_tokens
        )

    def log_cost(
        self,
        session_id: str,
        response: LLMResponse,
        query_type: str = "sub_llm",
        provider: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CostEntry:
        """Log cost for an LLM response.

        Args:
            session_id: Session identifier
            response: LLM response with usage info
            query_type: Type of query ("root" or "sub_llm")
            provider: Provider name (auto-detected if None)
            metadata: Additional metadata

        Returns:
            CostEntry that was logged
        """
        if not self.enabled:
            return CostEntry(
                timestamp=datetime.utcnow(),
                session_id=session_id,
                provider=provider or "unknown",
                model=response.model,
                query_type=query_type,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
            )

        # Extract usage info
        usage = response.usage or {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        # Auto-detect provider from model
        if provider is None:
            provider = self._registry.get_provider_for_model(response.model)
            if provider is None:
                provider = "unknown"

        # Calculate cost
        cost = self.calculate_cost(
            response.model, prompt_tokens, completion_tokens
        )

        # Create entry
        entry = CostEntry(
            timestamp=datetime.utcnow(),
            session_id=session_id,
            provider=provider,
            model=response.model,
            query_type=query_type,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            metadata=metadata or {},
        )

        # Store in memory
        self._entries.append(entry)

        # Write to log file
        if self._log_file:
            try:
                with open(self._log_file, "a") as f:
                    f.write(json.dumps(entry.to_dict()) + "\n")
            except Exception as e:
                logger.error("failed_to_write_cost_log", error=str(e))

        logger.debug(
            "cost_logged",
            session_id=session_id,
            model=response.model,
            cost_usd=cost,
            tokens=total_tokens,
        )

        return entry

    def get_session_cost(
        self,
        session_id: str,
        include_entries: bool = True,
    ) -> CostReport:
        """Get cost report for a session.

        Args:
            session_id: Session identifier
            include_entries: Whether to include detailed entries

        Returns:
            CostReport with breakdowns
        """
        entries = [
            e for e in self._entries if e.session_id == session_id
        ]

        return self._aggregate_costs(entries, include_entries)

    def get_total_cost(
        self,
        include_entries: bool = False,
    ) -> CostReport:
        """Get total cost across all sessions.

        Args:
            include_entries: Whether to include detailed entries

        Returns:
            CostReport with breakdowns
        """
        return self._aggregate_costs(self._entries, include_entries)

    def _aggregate_costs(
        self,
        entries: List[CostEntry],
        include_entries: bool = True,
    ) -> CostReport:
        """Aggregate costs from entries."""
        total_cost = sum(e.cost_usd for e in entries)
        total_tokens = sum(e.total_tokens for e in entries)

        by_provider: Dict[str, float] = {}
        by_model: Dict[str, float] = {}
        by_query_type: Dict[str, float] = {}

        for entry in entries:
            by_provider[entry.provider] = (
                by_provider.get(entry.provider, 0) + entry.cost_usd
            )
            by_model[entry.model] = (
                by_model.get(entry.model, 0) + entry.cost_usd
            )
            by_query_type[entry.query_type] = (
                by_query_type.get(entry.query_type, 0) + entry.cost_usd
            )

        return CostReport(
            total_cost=total_cost,
            total_tokens=total_tokens,
            total_requests=len(entries),
            by_provider=by_provider,
            by_model=by_model,
            by_query_type=by_query_type,
            entries=entries if include_entries else [],
        )

    def export_to_json(self, filepath: str) -> None:
        """Export all cost entries to JSON file.

        Args:
            filepath: Output file path
        """
        report = self.get_total_cost(include_entries=True)

        with open(filepath, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(
            "costs_exported",
            filepath=filepath,
            total_cost=report.total_cost,
            total_requests=report.total_requests,
        )

    def export_to_csv(self, filepath: str) -> None:
        """Export cost entries to CSV file.

        Args:
            filepath: Output file path
        """
        import csv

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "session_id",
                    "provider",
                    "model",
                    "query_type",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "cost_usd",
                ]
            )

            for entry in self._entries:
                writer.writerow(
                    [
                        entry.timestamp.isoformat(),
                        entry.session_id,
                        entry.provider,
                        entry.model,
                        entry.query_type,
                        entry.prompt_tokens,
                        entry.completion_tokens,
                        entry.total_tokens,
                        entry.cost_usd,
                    ]
                )

        logger.info(
            "costs_exported_csv",
            filepath=filepath,
            total_entries=len(self._entries),
        )

    def clear_session(self, session_id: str) -> None:
        """Clear cost entries for a session.

        Args:
            session_id: Session identifier
        """
        self._entries = [
            e for e in self._entries if e.session_id != session_id
        ]

    def clear_all(self) -> None:
        """Clear all cost entries."""
        self._entries.clear()
        logger.info("cost_tracker_cleared")
