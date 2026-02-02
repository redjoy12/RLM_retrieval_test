"""Query batcher for intelligent grouping of sub-LLM calls."""

from typing import Any, Dict, List

import structlog

from rlm.llm.sub_llm_manager import SubLLMCall

logger = structlog.get_logger()


class QueryBatcher:
    """Groups similar sub-LLM queries for efficient batch processing.

    This class analyzes sub-LLM calls and groups them by similarity to
    enable more efficient processing. Similar queries can be batched
    together for better API utilization and reduced costs.

    Features:
    - Keyword-based grouping for similar queries
    - Priority-based batch ordering
    - Configurable batch sizes
    - Dynamic grouping strategies

    Example:
        ```python
        batcher = QueryBatcher(similarity_threshold=0.8)

        calls = [
            SubLLMCall(query="What is AI?", priority=5),
            SubLLMCall(query="What is machine learning?", priority=3),
            SubLLMCall(query="What is the weather?", priority=5),
        ]

        groups = batcher.group_similar(calls)
        # Groups might be: {"ai": [call1, call2], "weather": [call3]}
        ```
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        max_batch_size: int = 10,
        enable_semantic_grouping: bool = False,
    ) -> None:
        """Initialize the query batcher.

        Args:
            similarity_threshold: Minimum similarity score (0-1) for grouping
            max_batch_size: Maximum calls per batch
            enable_semantic_grouping: Whether to use semantic similarity
                                     (Phase 2 feature, currently keyword-based)
        """
        self.similarity_threshold = similarity_threshold
        self.max_batch_size = max_batch_size
        self.enable_semantic_grouping = enable_semantic_grouping

        # Common keywords for grouping
        self._topic_keywords: Dict[str, List[str]] = {
            "definition": ["what is", "define", "meaning of", "explain"],
            "analysis": ["analyze", "compare", "contrast", "difference between"],
            "factual": ["when", "where", "who", "how many", "how much"],
            "causal": ["why", "how does", "what causes", "reason for"],
            "summary": ["summarize", "main points", "key findings", "overview"],
        }

        logger.info(
            "query_batcher_initialized",
            similarity_threshold=similarity_threshold,
            max_batch_size=max_batch_size,
            semantic_grouping=enable_semantic_grouping,
        )

    def group_similar(self, calls: List[SubLLMCall]) -> Dict[str, List[SubLLMCall]]:
        """Group similar calls together based on query content.

        Args:
            calls: List of SubLLMCall objects to group

        Returns:
            Dictionary mapping group IDs to lists of calls
        """
        if not calls:
            return {}

        groups: Dict[str, List[SubLLMCall]] = {}

        for call in calls:
            group_id = self._determine_group(call)

            if group_id not in groups:
                groups[group_id] = []

            groups[group_id].append(call)

        # Split large groups
        final_groups: Dict[str, List[SubLLMCall]] = {}
        group_counter = 0

        for group_id, group_calls in groups.items():
            if len(group_calls) <= self.max_batch_size:
                final_groups[f"{group_id}_{group_counter}"] = group_calls
                group_counter += 1
            else:
                # Split into smaller batches
                for i in range(0, len(group_calls), self.max_batch_size):
                    batch = group_calls[i : i + self.max_batch_size]
                    final_groups[f"{group_id}_{group_counter}"] = batch
                    group_counter += 1

        logger.debug(
            "calls_grouped",
            total_calls=len(calls),
            num_groups=len(final_groups),
            avg_group_size=len(calls) / len(final_groups) if final_groups else 0,
        )

        return final_groups

    def _determine_group(self, call: SubLLMCall) -> str:
        """Determine which group a call belongs to.

        Args:
            call: SubLLMCall to categorize

        Returns:
            Group identifier string
        """
        query_lower = call.query.lower()

        # Check for topic keywords
        for topic, keywords in self._topic_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return topic

        # Check if context hints at a topic
        if call.context:
            context_lower = call.context.lower()
            for topic, keywords in self._topic_keywords.items():
                for keyword in keywords:
                    if keyword in context_lower[:200]:  # Check first 200 chars
                        return f"{topic}_ctx"

        # Default to "mixed" group
        return "mixed"

    def group_by_priority(
        self,
        calls: List[SubLLMCall],
        priority_threshold: int = 5,
    ) -> Dict[str, List[SubLLMCall]]:
        """Group calls by priority levels.

        Args:
            calls: List of SubLLMCall objects
            priority_threshold: Priority level to split at (1-10)

        Returns:
            Dictionary with "high" and "low" priority groups
        """
        high_priority = [c for c in calls if c.priority <= priority_threshold]
        low_priority = [c for c in calls if c.priority > priority_threshold]

        groups: Dict[str, List[SubLLMCall]] = {}

        if high_priority:
            groups["high_priority"] = high_priority
        if low_priority:
            groups["low_priority"] = low_priority

        return groups

    def create_execution_order(
        self,
        groups: Dict[str, List[SubLLMCall]],
        strategy: str = "priority",
    ) -> List[str]:
        """Create optimal execution order for groups.

        Args:
            groups: Dictionary of grouped calls
            strategy: Ordering strategy ("priority", "size", "mixed")

        Returns:
            List of group IDs in execution order
        """
        group_ids = list(groups.keys())

        if strategy == "priority":
            # Execute high priority groups first
            def priority_key(gid: str) -> tuple:
                calls = groups[gid]
                avg_priority = sum(c.priority for c in calls) / len(calls)
                return (avg_priority, len(calls))  # Lower priority first, then size

            return sorted(group_ids, key=priority_key)

        elif strategy == "size":
            # Execute smaller groups first for quick wins
            return sorted(group_ids, key=lambda g: len(groups[g]))

        elif strategy == "mixed":
            # Interleave high priority and small groups
            priority_order = self.create_execution_order(groups, "priority")
            size_order = self.create_execution_order(groups, "size")

            # Alternate between strategies
            result = []
            p_idx, s_idx = 0, 0
            used = set()

            while len(result) < len(group_ids):
                # Add from priority order
                if p_idx < len(priority_order):
                    gid = priority_order[p_idx]
                    if gid not in used:
                        result.append(gid)
                        used.add(gid)
                    p_idx += 1

                # Add from size order
                if s_idx < len(size_order):
                    gid = size_order[s_idx]
                    if gid not in used:
                        result.append(gid)
                        used.add(gid)
                    s_idx += 1

            return result

        else:
            # Default: return as-is
            return group_ids

    def estimate_batch_efficiency(
        self,
        calls: List[SubLLMCall],
        groups: Dict[str, List[SubLLMCall]],
    ) -> Dict[str, Any]:
        """Estimate the efficiency of batching.

        Args:
            calls: Original list of calls
            groups: Grouped calls

        Returns:
            Dictionary with efficiency metrics
        """
        total_calls = len(calls)
        num_groups = len(groups)

        if num_groups == 0:
            return {"efficiency_score": 0.0, "avg_group_size": 0.0}

        avg_group_size = total_calls / num_groups

        # Efficiency score: higher is better
        # Ideal: all calls in one group (score = 1.0)
        # Worst: each call in its own group (score = 0.0)
        efficiency_score = 1.0 - (num_groups - 1) / total_calls if total_calls > 0 else 0.0

        return {
            "total_calls": total_calls,
            "num_groups": num_groups,
            "avg_group_size": avg_group_size,
            "efficiency_score": max(0.0, min(1.0, efficiency_score)),
            "max_group_size": max(len(g) for g in groups.values()),
            "min_group_size": min(len(g) for g in groups.values()),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get batcher configuration stats.

        Returns:
            Dictionary with batcher settings
        """
        return {
            "similarity_threshold": self.similarity_threshold,
            "max_batch_size": self.max_batch_size,
            "semantic_grouping_enabled": self.enable_semantic_grouping,
            "topic_keywords_defined": len(self._topic_keywords),
        }
