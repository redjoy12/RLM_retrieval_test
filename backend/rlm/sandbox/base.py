"""Sandbox interface for code execution."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from rlm.types import REPLResult


class REPLSandboxInterface(ABC):
    """Abstract interface for REPL sandbox environments.
    
    Implementations provide safe code execution with varying levels
    of isolation (local process, Docker container, cloud sandbox, etc.)
    """
    
    @abstractmethod
    async def execute(
        self,
        code: str,
        context: Any,
        sub_llm_callback: Callable[[str, Optional[str]], str],
        timeout: Optional[int] = None,
    ) -> REPLResult:
        """Execute Python code in the sandbox.
        
        Args:
            code: Python code to execute
            context: Document context object (ChunkedContext)
            sub_llm_callback: Function to call for sub-LLM queries
            timeout: Execution timeout in seconds (overrides default)
            
        Returns:
            REPLResult with output, errors, and sub-LLM calls
        """
        ...
    
    @abstractmethod
    def get_sandbox_type(self) -> str:
        """Get the type of sandbox.
        
        Returns:
            Sandbox type identifier (e.g., 'local', 'docker', 'modal')
        """
        ...
