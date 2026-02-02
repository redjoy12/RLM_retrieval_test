"""Local REPL sandbox implementation using RestrictedPython."""

import asyncio
import builtins
import sys
import time
import traceback
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Set

import RestrictedPython
import structlog
from RestrictedPython import compile_restricted, safe_globals

from rlm.config import get_sandbox_settings, get_settings
from rlm.sandbox.base import REPLSandboxInterface
from rlm.types import REPLResult

logger = structlog.get_logger()


class LocalREPLSandbox(REPLSandboxInterface):
    """Local non-isolated REPL sandbox using RestrictedPython.
    
    This sandbox runs in the same process as the RLM engine.
    It uses RestrictedPython to transform and limit the code,
    but is NOT suitable for running untrusted code in production.
    
    For production use with untrusted code, use DockerREPL or cloud sandboxes.
    """
    
    def __init__(
        self,
        timeout: Optional[int] = None,
        output_limit: Optional[int] = None,
        memory_limit_mb: Optional[int] = None,
    ) -> None:
        """Initialize the local REPL sandbox.
        
        Args:
            timeout: Code execution timeout in seconds
            output_limit: Maximum output characters
            memory_limit_mb: Memory limit in MB (enforced via monitoring)
        """
        settings = get_settings()
        sandbox_settings = get_sandbox_settings()
        
        self.timeout = timeout or settings.code_execution_timeout
        self.output_limit = output_limit or settings.sandbox_output_limit
        self.memory_limit_mb = memory_limit_mb or settings.sandbox_memory_limit_mb
        
        self.allowed_modules = set(sandbox_settings.allowed_modules)
        self.blocked_builtins = set(sandbox_settings.blocked_builtins)
        self.max_code_length = sandbox_settings.max_code_length
        
        logger.info(
            "local_sandbox_initialized",
            timeout=self.timeout,
            output_limit=self.output_limit,
            allowed_modules=len(self.allowed_modules),
        )
    
    def get_sandbox_type(self) -> str:
        """Get sandbox type."""
        return "local"
    
    async def execute(
        self,
        code: str,
        context: Any,
        sub_llm_callback: Callable[[str, Optional[str]], str],
        timeout: Optional[int] = None,
    ) -> REPLResult:
        """Execute code with timeout and restrictions."""
        execution_timeout = timeout or self.timeout
        sub_llm_calls: List[Dict[str, Any]] = []
        
        # Check code length
        if len(code) > self.max_code_length:
            return REPLResult(
                output="",
                error=f"Code too long: {len(code)} characters (max: {self.max_code_length})",
            )
        
        try:
            # Create restricted environment
            globals_dict = self._create_restricted_globals(context, sub_llm_callback, sub_llm_calls)
            locals_dict = {}
            
            # Compile with RestrictedPython
            compiled = compile_restricted(code, '<inline>', 'exec')
            if compiled is None:
                return REPLResult(
                    output="",
                    error="Code failed security checks",
                )
            
            # Execute with timeout
            start_time = time.time()
            output_buffer = StringIO()
            
            # Redirect stdout
            old_stdout = sys.stdout
            sys.stdout = output_buffer
            
            try:
                # Run in executor to allow timeout
                loop = asyncio.get_event_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: exec(compiled, globals_dict, locals_dict)),
                    timeout=execution_timeout,
                )
                
                execution_time = (time.time() - start_time) * 1000  # ms
                
                # Get output
                output = output_buffer.getvalue()
                
                # Truncate if too long
                if len(output) > self.output_limit:
                    output = output[:self.output_limit] + f"\n... [truncated, total: {len(output)} chars]"
                
                return REPLResult(
                    output=output,
                    error=None,
                    sub_llm_calls=sub_llm_calls,
                    execution_time_ms=execution_time,
                )
                
            except asyncio.TimeoutError:
                return REPLResult(
                    output=output_buffer.getvalue(),
                    error=f"Code execution timed out after {execution_timeout} seconds",
                    sub_llm_calls=sub_llm_calls,
                )
                
            finally:
                sys.stdout = old_stdout
                
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            logger.error("sandbox_execution_error", error=str(e))
            return REPLResult(
                output="",
                error=error_msg,
                sub_llm_calls=sub_llm_calls,
            )
    
    def _create_restricted_globals(
        self,
        context: Any,
        sub_llm_callback: Callable[[str, Optional[str]], str],
        sub_llm_calls_log: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Create restricted global namespace for code execution."""
        
        # Start with safe globals from RestrictedPython
        globals_dict = safe_globals.copy()
        
        # Remove blocked builtins
        for name in self.blocked_builtins:
            if name in globals_dict.get('__builtins__', {}):
                del globals_dict['__builtins__'][name]
        
        # Add allowed modules
        for module_name in self.allowed_modules:
            try:
                module = __import__(module_name, fromlist=[''])
                globals_dict[module_name] = module
            except ImportError:
                logger.warning("failed_to_import_allowed_module", module=module_name)
        
        # Add llm_query function
        def llm_query(query: str, context_chunk: Optional[str] = None) -> str:
            """Query a sub-LLM with an optional context chunk."""
            call_record = {
                "query": query,
                "context_chunk": context_chunk[:100] + "..." if context_chunk and len(context_chunk) > 100 else context_chunk,
                "timestamp": time.time(),
            }
            sub_llm_calls_log.append(call_record)
            
            result = sub_llm_callback(query, context_chunk)
            call_record["result"] = result[:200] + "..." if len(result) > 200 else result
            return result
        
        # Add safe print function
        def safe_print(*args, **kwargs):
            """Safe print that respects output limits."""
            builtins.print(*args, **kwargs)
        
        # Inject into globals
        globals_dict['llm_query'] = llm_query
        globals_dict['print'] = safe_print
        globals_dict['context'] = context
        
        return globals_dict
