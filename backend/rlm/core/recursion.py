"""Recursion controller for managing sub-LLM call depth and limits."""

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import structlog

from rlm.config import get_settings

logger = structlog.get_logger()


@dataclass
class RecursionNode:
    """Represents a node in the recursion tree."""
    
    session_id: str
    parent_id: Optional[str]
    depth: int
    query: str
    children: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=lambda: __import__('time').time())


class RecursionController:
    """Manages recursion depth and call limits for RLM execution.
    
    Tracks:
    - Recursion depth (how many levels deep)
    - Total call count (across all branches)
    - Call tree structure for visualization
    """
    
    def __init__(
        self,
        max_depth: Optional[int] = None,
        max_total_calls: Optional[int] = None,
    ) -> None:
        """Initialize the recursion controller.
        
        Args:
            max_depth: Maximum recursion depth (default from settings)
            max_total_calls: Maximum total sub-LLM calls (default from settings)
        """
        settings = get_settings()
        
        self.max_depth = max_depth or settings.max_recursion_depth
        self.max_total_calls = max_total_calls or settings.max_sub_llm_calls
        
        self._call_count = 0
        self._nodes: Dict[str, RecursionNode] = {}
        self._root_session_id: Optional[str] = None
        
        logger.info(
            "recursion_controller_initialized",
            max_depth=self.max_depth,
            max_calls=self.max_total_calls,
        )
    
    def initialize_root(self, query: str) -> str:
        """Initialize a new root session.
        
        Args:
            query: The initial query
            
        Returns:
            Root session ID
        """
        session_id = str(uuid.uuid4())
        self._root_session_id = session_id
        
        node = RecursionNode(
            session_id=session_id,
            parent_id=None,
            depth=0,
            query=query,
        )
        self._nodes[session_id] = node
        
        logger.info("root_session_initialized", session_id=session_id)
        return session_id
    
    def can_spawn_sub_llm(self, parent_session_id: str) -> bool:
        """Check if we can spawn another sub-LLM call.
        
        Args:
            parent_session_id: ID of the parent session
            
        Returns:
            True if allowed, False if limits reached
        """
        # Check total call limit
        if self._call_count >= self.max_total_calls:
            logger.warning(
                "total_call_limit_reached",
                current=self._call_count,
                max=self.max_total_calls,
            )
            return False
        
        # Check depth limit
        parent_node = self._nodes.get(parent_session_id)
        if not parent_node:
            logger.error("parent_session_not_found", parent_id=parent_session_id)
            return False
        
        if parent_node.depth >= self.max_depth - 1:
            logger.warning(
                "depth_limit_reached",
                current_depth=parent_node.depth,
                max_depth=self.max_depth,
                parent_id=parent_session_id,
            )
            return False
        
        return True
    
    def enter_sub_llm(
        self,
        parent_session_id: str,
        query: str,
    ) -> Optional[str]:
        """Enter a new sub-LLM context.
        
        Args:
            parent_session_id: ID of the parent session
            query: The sub-LLM query
            
        Returns:
            New session ID if allowed, None if limits reached
        """
        if not self.can_spawn_sub_llm(parent_session_id):
            return None
        
        parent_node = self._nodes[parent_session_id]
        
        # Create new node
        new_session_id = str(uuid.uuid4())
        new_node = RecursionNode(
            session_id=new_session_id,
            parent_id=parent_session_id,
            depth=parent_node.depth + 1,
            query=query,
        )
        
        self._nodes[new_session_id] = new_node
        parent_node.children.append(new_session_id)
        self._call_count += 1
        
        logger.debug(
            "sub_llm_entered",
            session_id=new_session_id,
            parent_id=parent_session_id,
            depth=new_node.depth,
            total_calls=self._call_count,
        )
        
        return new_session_id
    
    def exit_sub_llm(self, session_id: str) -> None:
        """Exit a sub-LLM context (cleanup if needed).
        
        Args:
            session_id: Session ID to exit
        """
        # Currently just logging, could do cleanup in future
        if session_id in self._nodes:
            node = self._nodes[session_id]
            logger.debug(
                "sub_llm_exited",
                session_id=session_id,
                depth=node.depth,
                duration_seconds=__import__('time').time() - node.created_at,
            )
    
    def get_call_tree(self) -> Dict[str, any]:
        """Get the full call tree structure.
        
        Returns:
            Dictionary representing the call tree
        """
        if not self._root_session_id:
            return {}
        
        def build_tree(node_id: str) -> Dict:
            node = self._nodes[node_id]
            return {
                "session_id": node.session_id,
                "depth": node.depth,
                "query": node.query,
                "children": [build_tree(child_id) for child_id in node.children],
            }
        
        return build_tree(self._root_session_id)
    
    def get_stats(self) -> Dict[str, any]:
        """Get recursion statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "total_calls": self._call_count,
            "max_depth": self.max_depth,
            "max_calls": self.max_total_calls,
            "current_depth": max(
                (node.depth for node in self._nodes.values()),
                default=0,
            ),
            "active_nodes": len(self._nodes),
        }
    
    def get_current_depth(self, session_id: str) -> int:
        """Get the current recursion depth for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Current depth (0 for root)
        """
        node = self._nodes.get(session_id)
        return node.depth if node else 0
