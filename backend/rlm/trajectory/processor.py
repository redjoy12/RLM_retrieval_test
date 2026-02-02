"""Trajectory Data Processor

Converts flat JSONL trajectory logs into visualization-friendly structures.
Supports tree conversion, timeline generation, cost analysis, and statistics.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import json

import structlog

logger = structlog.get_logger()


class TrajectoryStepType(str, Enum):
    """Types of steps in a trajectory."""
    ROOT_LLM_START = "ROOT_LLM_START"
    ROOT_LLM_COMPLETE = "ROOT_LLM_COMPLETE"
    CODE_EXECUTION_START = "CODE_EXECUTION_START"
    CODE_EXECUTION_COMPLETE = "CODE_EXECUTION_COMPLETE"
    SUB_LLM_SPAWN = "SUB_LLM_SPAWN"
    SUB_LLM_COMPLETE = "SUB_LLM_COMPLETE"
    RECURSION_LIMIT_HIT = "RECURSION_LIMIT_HIT"
    ERROR = "ERROR"
    FINAL_ANSWER = "FINAL_ANSWER"


@dataclass
class TokenCost:
    """Token usage and cost information."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class TrajectoryNode:
    """Represents a node in the trajectory tree.
    
    Attributes:
        id: Unique node identifier
        type: Type of step
        parent_id: Parent node ID (None for root)
        children: List of child node IDs
        timestamp: When the step occurred
        duration_ms: Duration of the step in milliseconds
        depth: Recursion depth (0 for root)
        data: Additional step-specific data
        cost: Token usage and cost information
    """
    id: str
    type: TrajectoryStepType
    parent_id: Optional[str]
    children: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: Optional[float] = None
    depth: int = 0
    data: Dict[str, Any] = field(default_factory=dict)
    cost: TokenCost = field(default_factory=TokenCost)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "parent_id": self.parent_id,
            "children": self.children,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "depth": self.depth,
            "data": self.data,
            "cost": {
                "input_tokens": self.cost.input_tokens,
                "output_tokens": self.cost.output_tokens,
                "total_tokens": self.cost.total_tokens,
                "cost_usd": self.cost.cost_usd,
            },
        }


@dataclass
class TrajectoryTree:
    """Complete tree structure for visualization.
    
    Attributes:
        root_id: ID of the root node
        nodes: Dictionary mapping node IDs to TrajectoryNode objects
        total_nodes: Total number of nodes in the tree
        max_depth: Maximum recursion depth
        total_duration_ms: Total execution time
        total_cost_usd: Total cost in USD
        session_id: Session identifier
    """
    root_id: str
    nodes: Dict[str, TrajectoryNode]
    total_nodes: int = 0
    max_depth: int = 0
    total_duration_ms: float = 0.0
    total_cost_usd: float = 0.0
    session_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tree to dictionary."""
        return {
            "root_id": self.root_id,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "total_nodes": self.total_nodes,
            "max_depth": self.max_depth,
            "total_duration_ms": self.total_duration_ms,
            "total_cost_usd": self.total_cost_usd,
            "session_id": self.session_id,
        }


@dataclass
class TimelineEvent:
    """Represents an event in the timeline view."""
    node_id: str
    type: TrajectoryStepType
    start_time: datetime
    end_time: Optional[datetime]
    duration_ms: Optional[float]
    depth: int
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "type": self.type.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "depth": self.depth,
            "data": self.data,
        }


@dataclass
class CostBreakdown:
    """Cost breakdown by category.
    
    Attributes:
        by_depth: Cost aggregated by recursion depth
        by_type: Cost aggregated by step type
        total_input_tokens: Total input tokens used
        total_output_tokens: Total output tokens used
        total_tokens: Total tokens used
        total_cost_usd: Total cost in USD
    """
    by_depth: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    by_type: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "by_depth": self.by_depth,
            "by_type": self.by_type,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
        }


@dataclass
class ExecutionStats:
    """Execution statistics.
    
    Attributes:
        total_steps: Total number of steps
        total_llm_calls: Total LLM calls (root + sub)
        total_code_executions: Total code execution blocks
        total_errors: Total errors encountered
        max_recursion_depth: Maximum recursion depth reached
        total_duration_ms: Total execution time
        avg_step_duration_ms: Average step duration
    """
    total_steps: int = 0
    total_llm_calls: int = 0
    total_code_executions: int = 0
    total_errors: int = 0
    max_recursion_depth: int = 0
    total_duration_ms: float = 0.0
    avg_step_duration_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_steps": self.total_steps,
            "total_llm_calls": self.total_llm_calls,
            "total_code_executions": self.total_code_executions,
            "total_errors": self.total_errors,
            "max_recursion_depth": self.max_recursion_depth,
            "total_duration_ms": self.total_duration_ms,
            "avg_step_duration_ms": self.avg_step_duration_ms,
        }


class TrajectoryProcessor:
    """Process raw trajectory logs into visualization formats.
    
    This class converts flat JSONL trajectory logs into hierarchical tree
    structures, timeline data, cost breakdowns, and execution statistics.
    
    Example:
        >>> processor = TrajectoryProcessor(Path("./logs"))
        >>> tree = processor.to_tree("session-123")
        >>> timeline = processor.to_timeline("session-123")
        >>> costs = processor.get_cost_breakdown("session-123")
    """
    
    # Cost per 1K tokens (approximate, should match your LLM pricing)
    COST_PER_1K_INPUT_TOKENS = 0.0015  # GPT-4o-mini pricing
    COST_PER_1K_OUTPUT_TOKENS = 0.006
    
    def __init__(self, log_dir: Path) -> None:
        """Initialize the trajectory processor.
        
        Args:
            log_dir: Directory containing trajectory JSONL files
        """
        self.log_dir = Path(log_dir)
        self._cache: Dict[str, List[Dict]] = {}
        
        logger.info(
            "trajectory_processor_initialized",
            log_dir=str(self.log_dir),
        )
    
    def _load_trajectory(self, session_id: str) -> List[Dict[str, Any]]:
        """Load trajectory data from JSONL file.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of trajectory steps
        """
        if session_id in self._cache:
            return self._cache[session_id]
        
        log_file = self.log_dir / f"{session_id}.jsonl"
        
        if not log_file.exists():
            logger.warning(
                "trajectory_file_not_found",
                session_id=session_id,
                log_file=str(log_file),
            )
            return []
        
        trajectory = []
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        trajectory.append(json.loads(line))
            
            self._cache[session_id] = trajectory
            
            logger.debug(
                "trajectory_loaded",
                session_id=session_id,
                step_count=len(trajectory),
            )
            
        except Exception as e:
            logger.error(
                "failed_to_load_trajectory",
                session_id=session_id,
                error=str(e),
            )
        
        return trajectory
    
    def to_tree(self, session_id: str) -> TrajectoryTree:
        """Convert trajectory JSONL to tree structure.
        
        Builds a hierarchical tree from flat trajectory steps, tracking
        parent-child relationships based on session_id prefixes.
        
        Args:
            session_id: Session identifier
            
        Returns:
            TrajectoryTree with hierarchical structure
        """
        steps = self._load_trajectory(session_id)
        
        if not steps:
            return TrajectoryTree(
                root_id=session_id,
                nodes={},
                session_id=session_id,
            )
        
        nodes: Dict[str, TrajectoryNode] = {}
        root_id = session_id
        max_depth = 0
        total_cost = 0.0
        
        # First pass: create all nodes
        for step in steps:
            step_session_id = step.get("session_id", session_id)
            step_type_str = step.get("step_type", "ERROR")
            
            try:
                step_type = TrajectoryStepType(step_type_str)
            except ValueError:
                step_type = TrajectoryStepType.ERROR
            
            # Calculate depth from session_id
            # Root: session-123
            # Sub: session-123-sub-1
            # Sub-sub: session-123-sub-1-sub-2
            depth = step_session_id.count("-sub-")
            max_depth = max(max_depth, depth)
            
            # Determine parent
            parent_id = None
            if "-sub-" in step_session_id:
                # Remove last -sub-X to get parent
                parts = step_session_id.rsplit("-sub-", 1)
                if len(parts) == 2:
                    parent_id = parts[0]
            
            # Extract cost information from usage data
            data = step.get("data", {})
            usage = data.get("usage", {})
            input_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
            output_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
            
            # Calculate cost
            input_cost = (input_tokens / 1000) * self.COST_PER_1K_INPUT_TOKENS
            output_cost = (output_tokens / 1000) * self.COST_PER_1K_OUTPUT_TOKENS
            cost_usd = input_cost + output_cost
            total_cost += cost_usd
            
            # Parse timestamp
            timestamp_str = step.get("timestamp", datetime.utcnow().isoformat())
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except:
                timestamp = datetime.utcnow()
            
            node = TrajectoryNode(
                id=step_session_id,
                type=step_type,
                parent_id=parent_id,
                timestamp=timestamp,
                depth=depth,
                data=data,
                cost=TokenCost(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    cost_usd=cost_usd,
                ),
            )
            
            nodes[step_session_id] = node
        
        # Second pass: build parent-child relationships
        for node_id, node in nodes.items():
            if node.parent_id and node.parent_id in nodes:
                nodes[node.parent_id].children.append(node_id)
        
        # Calculate durations
        self._calculate_durations(nodes, steps)
        
        # Calculate total duration from root
        total_duration = 0.0
        if root_id in nodes:
            root_node = nodes[root_id]
            if root_node.children:
                # Find last child completion
                last_child_id = root_node.children[-1]
                if last_child_id in nodes:
                    last_child = nodes[last_child_id]
                    if last_child.duration_ms:
                        last_end = last_child.timestamp.timestamp() * 1000 + last_child.duration_ms
                        root_start = root_node.timestamp.timestamp() * 1000
                        total_duration = last_end - root_start
        
        return TrajectoryTree(
            root_id=root_id,
            nodes=nodes,
            total_nodes=len(nodes),
            max_depth=max_depth,
            total_duration_ms=total_duration,
            total_cost_usd=total_cost,
            session_id=session_id,
        )
    
    def _calculate_durations(
        self,
        nodes: Dict[str, TrajectoryNode],
        steps: List[Dict],
    ) -> None:
        """Calculate durations for each node based on step pairs.
        
        Matches start/complete pairs to calculate durations.
        """
        # Find pairs of start/complete steps
        start_times: Dict[str, datetime] = {}
        
        for step in steps:
            session_id = step.get("session_id", "")
            step_type = step.get("step_type", "")
            timestamp_str = step.get("timestamp", "")
            
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except:
                continue
            
            if "START" in step_type or step_type == "SUB_LLM_SPAWN":
                start_times[session_id] = timestamp
            elif "COMPLETE" in step_type or step_type == "SUB_LLM_COMPLETE":
                if session_id in start_times:
                    start_time = start_times[session_id]
                    duration_ms = (timestamp - start_time).total_seconds() * 1000
                    if session_id in nodes:
                        nodes[session_id].duration_ms = duration_ms
    
    def to_timeline(self, session_id: str) -> List[TimelineEvent]:
        """Convert trajectory to timeline format.
        
        Creates a Gantt-style timeline representation of the execution.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of timeline events sorted by start time
        """
        tree = self.to_tree(session_id)
        events: List[TimelineEvent] = []
        
        for node_id, node in tree.nodes.items():
            # Determine end time
            end_time = None
            if node.duration_ms:
                end_time = datetime.fromtimestamp(
                    node.timestamp.timestamp() + node.duration_ms / 1000
                )
            
            event = TimelineEvent(
                node_id=node_id,
                type=node.type,
                start_time=node.timestamp,
                end_time=end_time,
                duration_ms=node.duration_ms,
                depth=node.depth,
                data=node.data,
            )
            events.append(event)
        
        # Sort by start time
        events.sort(key=lambda e: e.start_time)
        
        return events
    
    def get_cost_breakdown(self, session_id: str) -> CostBreakdown:
        """Aggregate costs by depth and type.
        
        Args:
            session_id: Session identifier
            
        Returns:
            CostBreakdown with aggregated statistics
        """
        tree = self.to_tree(session_id)
        
        by_depth: Dict[int, Dict[str, Any]] = {}
        by_type: Dict[str, Dict[str, Any]] = {}
        
        total_input = 0
        total_output = 0
        total_tokens = 0
        total_cost = 0.0
        
        for node in tree.nodes.values():
            # Aggregate by depth
            depth = node.depth
            if depth not in by_depth:
                by_depth[depth] = {
                    "count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                }
            
            by_depth[depth]["count"] += 1
            by_depth[depth]["input_tokens"] += node.cost.input_tokens
            by_depth[depth]["output_tokens"] += node.cost.output_tokens
            by_depth[depth]["total_tokens"] += node.cost.total_tokens
            by_depth[depth]["cost_usd"] += node.cost.cost_usd
            
            # Aggregate by type
            type_name = node.type.value
            if type_name not in by_type:
                by_type[type_name] = {
                    "count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                }
            
            by_type[type_name]["count"] += 1
            by_type[type_name]["input_tokens"] += node.cost.input_tokens
            by_type[type_name]["output_tokens"] += node.cost.output_tokens
            by_type[type_name]["total_tokens"] += node.cost.total_tokens
            by_type[type_name]["cost_usd"] += node.cost.cost_usd
            
            # Totals
            total_input += node.cost.input_tokens
            total_output += node.cost.output_tokens
            total_tokens += node.cost.total_tokens
            total_cost += node.cost.cost_usd
        
        return CostBreakdown(
            by_depth=by_depth,
            by_type=by_type,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
        )
    
    def get_statistics(self, session_id: str) -> ExecutionStats:
        """Calculate execution statistics.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ExecutionStats with aggregated metrics
        """
        tree = self.to_tree(session_id)
        steps = self._load_trajectory(session_id)
        
        if not steps:
            return ExecutionStats()
        
        total_steps = len(steps)
        llm_calls = 0
        code_execs = 0
        errors = 0
        total_duration = 0.0
        
        durations: List[float] = []
        
        for step in steps:
            step_type = step.get("step_type", "")
            
            if "LLM" in step_type:
                llm_calls += 1
            elif "CODE" in step_type:
                code_execs += 1
            elif step_type == "ERROR":
                errors += 1
        
        for node in tree.nodes.values():
            if node.duration_ms:
                durations.append(node.duration_ms)
        
        if durations:
            total_duration = sum(durations)
            avg_duration = total_duration / len(durations)
        else:
            avg_duration = 0.0
        
        return ExecutionStats(
            total_steps=total_steps,
            total_llm_calls=llm_calls,
            total_code_executions=code_execs,
            total_errors=errors,
            max_recursion_depth=tree.max_depth,
            total_duration_ms=tree.total_duration_ms,
            avg_step_duration_ms=avg_duration,
        )
    
    def get_node_details(self, session_id: str, node_id: str) -> Optional[TrajectoryNode]:
        """Get details for a specific node.
        
        Args:
            session_id: Session identifier
            node_id: Node identifier
            
        Returns:
            TrajectoryNode if found, None otherwise
        """
        tree = self.to_tree(session_id)
        return tree.nodes.get(node_id)
    
    def clear_cache(self, session_id: Optional[str] = None) -> None:
        """Clear the trajectory cache.
        
        Args:
            session_id: Specific session to clear, or None to clear all
        """
        if session_id:
            self._cache.pop(session_id, None)
            logger.debug("trajectory_cache_cleared", session_id=session_id)
        else:
            self._cache.clear()
            logger.debug("trajectory_cache_cleared_all")
