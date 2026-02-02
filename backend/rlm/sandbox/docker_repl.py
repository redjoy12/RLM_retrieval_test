"""Docker-based REPL sandbox implementation for secure code execution."""

import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import structlog

from rlm.config import get_docker_settings, get_settings
from rlm.sandbox.base import REPLSandboxInterface
from rlm.sandbox.monitor import ResourceMonitor, ResourceUsageSummary
from rlm.sandbox.security import SecurityProfiles, SecurityValidator
from rlm.types import REPLResult

logger = structlog.get_logger()


class DockerREPLSandbox(REPLSandboxInterface):
    """Docker-based REPL sandbox for secure, isolated code execution.

    This sandbox runs code in an isolated Docker container with:
    - Resource limits (memory, CPU)
    - Security hardening (read-only, capability dropping)
    - Network isolation
    - Automatic cleanup

    Example:
        ```python
        sandbox = DockerREPLSandbox(
            image="python:3.11-slim",
            memory_limit="512m",
            cpu_limit=1.0,
            security_profile="strict"
        )

        result = await sandbox.execute(
            code="print('Hello from Docker!')",
            context=document_context,
            sub_llm_callback=mock_callback,
            timeout=30
        )
        ```

    Requirements:
        - Docker Engine installed and running
        - User has permissions to run containers
        - `docker` Python SDK installed
    """

    def __init__(
        self,
        image: Optional[str] = None,
        timeout: Optional[int] = None,
        memory_limit: Optional[str] = None,
        cpu_limit: Optional[float] = None,
        network_enabled: Optional[bool] = None,
        security_profile: Optional[str] = None,
        auto_cleanup: Optional[bool] = None,
    ) -> None:
        """Initialize the Docker REPL sandbox.

        Args:
            image: Docker image to use (default from settings)
            timeout: Execution timeout in seconds
            memory_limit: Memory limit (e.g., "512m", "1g")
            cpu_limit: CPU limit in cores (e.g., 1.0, 2.0)
            network_enabled: Whether to enable network access
            security_profile: Security profile ("strict", "standard", "development")
            auto_cleanup: Whether to auto-remove containers after execution
        """
        settings = get_settings()
        docker_settings = get_docker_settings()

        # Initialize configuration
        self.image = image or docker_settings.image
        self.timeout = timeout or settings.code_execution_timeout
        self.memory_limit = memory_limit or docker_settings.memory_limit
        self.cpu_limit = cpu_limit or docker_settings.cpu_limit
        self.network_enabled = network_enabled if network_enabled is not None else docker_settings.network_enabled
        self.security_profile_name = security_profile or docker_settings.security_profile
        self.auto_cleanup = auto_cleanup if auto_cleanup is not None else docker_settings.auto_cleanup

        # Get security profile
        self.security_profile = SecurityProfiles.get_profile(
            self.security_profile_name,
            memory_limit=self.memory_limit,
            cpu_limit=self.cpu_limit,
        )

        # Initialize Docker client (lazy loading)
        self._docker_client: Optional[Any] = None

        # Resource monitor
        self._monitor = ResourceMonitor()

        logger.info(
            "docker_sandbox_initialized",
            image=self.image,
            memory_limit=self.memory_limit,
            cpu_limit=self.cpu_limit,
            security_profile=self.security_profile_name,
        )

    def get_sandbox_type(self) -> str:
        """Get sandbox type."""
        return "docker"

    def _get_docker_client(self) -> Any:
        """Get or create Docker client."""
        if self._docker_client is None:
            import docker

            self._docker_client = docker.from_env()
        return self._docker_client

    async def execute(
        self,
        code: str,
        context: Any,
        sub_llm_callback: Callable[[str, Optional[str]], str],
        timeout: Optional[int] = None,
    ) -> REPLResult:
        """Execute Python code in a Docker container.

        Args:
            code: Python code to execute
            context: Document context (ChunkedContext)
            sub_llm_callback: Function for sub-LLM calls
            timeout: Execution timeout (overrides default)

        Returns:
            REPLResult with output, errors, and metadata
        """
        execution_timeout = timeout or self.timeout
        container_name = f"rlm-sandbox-{uuid.uuid4().hex[:8]}"
        sub_llm_calls: List[Dict[str, Any]] = []

        temp_dir = None
        container = None

        try:
            # Validate Docker is available
            is_installed, message = SecurityValidator.validate_docker_installation()
            if not is_installed:
                return REPLResult(
                    output="",
                    error=f"Docker not available: {message}",
                    sub_llm_calls=sub_llm_calls,
                )

            # Create temporary directory for code and context
            temp_dir = tempfile.mkdtemp(prefix="rlm-sandbox-")
            temp_path = Path(temp_dir)

            # Write code to file
            code_file = temp_path / "script.py"
            code_file.write_text(code, encoding="utf-8")

            # Write context to file (if provided)
            context_file = temp_path / "context.txt"
            if context and hasattr(context, "_content"):
                context_file.write_text(context._content, encoding="utf-8")
            elif context and isinstance(context, str):
                context_file.write_text(context, encoding="utf-8")

            # Build the wrapper script that will run in container
            wrapper_code = self._build_wrapper_code(code_file.name, context_file.name)

            # Write wrapper to file
            wrapper_file = temp_path / "wrapper.py"
            wrapper_file.write_text(wrapper_code, encoding="utf-8")

            # Get Docker client
            client = self._get_docker_client()

            # Pull image if needed
            try:
                client.images.get(self.image)
            except Exception:
                logger.info("docker_pulling_image", image=self.image)
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: client.images.pull(self.image)
                )

            # Prepare host configuration from security profile
            host_config = self.security_profile.to_docker_config()

            # Override network mode if specified
            if self.network_enabled:
                host_config["NetworkMode"] = "bridge"

            # Create and start container
            logger.debug("docker_creating_container", name=container_name)

            container = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.containers.create(
                    image=self.image,
                    name=container_name,
                    command=["python", "/sandbox/wrapper.py"],
                    volumes={
                        str(temp_path): {"bind": "/sandbox", "mode": "ro"},
                    },
                    working_dir="/sandbox",
                    host_config=host_config,
                    detach=True,
                )
            )

            # Start container and monitoring
            await asyncio.get_event_loop().run_in_executor(None, container.start)
            await self._monitor.start_monitoring(container)

            logger.info(
                "docker_container_started",
                container_id=container.short_id,
                name=container_name,
            )

            # Wait for container to finish with timeout
            try:
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: container.wait()
                    ),
                    timeout=execution_timeout,
                )

                exit_code = result.get("StatusCode", -1)

            except asyncio.TimeoutError:
                logger.warning(
                    "docker_execution_timeout",
                    container=container_name,
                    timeout=execution_timeout,
                )
                # Kill the container
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None, lambda: container.kill(signal="SIGKILL")
                    )
                except Exception:
                    pass

                return REPLResult(
                    output="",
                    error=f"Code execution timed out after {execution_timeout} seconds",
                    sub_llm_calls=sub_llm_calls,
                )

            # Get logs
            logs = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: container.logs(stdout=True, stderr=True, timestamps=False).decode("utf-8", errors="replace")
            )

            # Stop monitoring and get resource summary
            resource_summary = await self._monitor.stop_monitoring()

            # Parse output and errors
            output, error = self._parse_output(logs, exit_code)

            # Cleanup container
            if self.auto_cleanup and container:
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None, lambda: container.remove(force=True)
                    )
                except Exception as e:
                    logger.warning("docker_cleanup_error", error=str(e))

            logger.info(
                "docker_execution_complete",
                container=container_name,
                exit_code=exit_code,
                peak_memory_mb=resource_summary.peak_memory_mb,
                execution_time_ms=resource_summary.execution_time_ms,
            )

            return REPLResult(
                output=output,
                error=error,
                sub_llm_calls=sub_llm_calls,
                execution_time_ms=resource_summary.execution_time_ms,
                memory_usage_mb=resource_summary.peak_memory_mb,
            )

        except Exception as e:
            logger.error("docker_execution_error", error=str(e))
            return REPLResult(
                output="",
                error=f"Docker execution failed: {str(e)}",
                sub_llm_calls=sub_llm_calls,
            )

        finally:
            # Cleanup temporary directory
            if temp_dir:
                try:
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception:
                    pass

            # Ensure container is cleaned up
            if container and self.auto_cleanup:
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None, lambda: container.remove(force=True)
                    )
                except Exception:
                    pass

    def _build_wrapper_code(self, code_file: str, context_file: str) -> str:
        """Build the wrapper script that runs inside the container.

        This script sets up a restricted environment and executes user code.

        Args:
            code_file: Name of the code file in container
            context_file: Name of the context file in container

        Returns:
            Wrapper script content
        """
        return '''
import sys
import json
import traceback
from io import StringIO

# Redirect stdout/stderr to capture output
old_stdout = sys.stdout
old_stderr = sys.stderr
sys.stdout = captured_output = StringIO()
sys.stderr = captured_errors = StringIO()

try:
    # Read context
    context = ""
    try:
        with open("/sandbox/{context_file}", "r", encoding="utf-8") as f:
            context = f.read()
    except Exception:
        pass

    # Mock llm_query function (real callback happens outside container)
    def llm_query(query: str, context_chunk: str = None) -> str:
        print(f"[SUB_LLM_CALL: {query[:50]}...]")
        return f"[Mock sub-LLM response for: {query[:50]}]"

    # Execute user code
    with open("/sandbox/{code_file}", "r", encoding="utf-8") as f:
        code = f.read()

    # Create restricted globals
    safe_globals = {{
        "__builtins__": {{
            "print": print,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "type": type,
            "isinstance": isinstance,
            "hasattr": hasattr,
            "getattr": getattr,
            "setattr": setattr,
            "Exception": Exception,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "IndexError": IndexError,
        }},
        "context": context,
        "llm_query": llm_query,
        "json": __import__("json"),
        "re": __import__("re"),
        "math": __import__("math"),
        "random": __import__("random"),
        "datetime": __import__("datetime"),
        "collections": __import__("collections"),
        "itertools": __import__("itertools"),
        "statistics": __import__("statistics"),
        "typing": __import__("typing"),
        "string": __import__("string"),
        "hashlib": __import__("hashlib"),
        "base64": __import__("base64"),
    }}

    exec(code, safe_globals, {{}})

    # Restore stdout/stderr and print captured output
    sys.stdout = old_stdout
    sys.stderr = old_stderr

    output = captured_output.getvalue()
    errors = captured_errors.getvalue()

    if output:
        print("===OUTPUT_START===")
        print(output)
        print("===OUTPUT_END===")

    if errors:
        print("===ERROR_START===", file=sys.stderr)
        print(errors, file=sys.stderr)
        print("===ERROR_END===", file=sys.stderr)

    sys.exit(0)

except Exception as e:
    sys.stdout = old_stdout
    sys.stderr = old_stderr

    print("===ERROR_START===", file=sys.stderr)
    print(f"Execution error: {{str(e)}}")
    print(traceback.format_exc())
    print("===ERROR_END===", file=sys.stderr)
    sys.exit(1)
'''.format(code_file=code_file, context_file=context_file)

    def _parse_output(self, logs: str, exit_code: int) -> tuple[str, Optional[str]]:
        """Parse container output to extract stdout and stderr.

        Args:
            logs: Raw container logs
            exit_code: Container exit code

        Returns:
            Tuple of (output, error)
        """
        output = ""
        error = None

        # Extract output between markers
        if "===OUTPUT_START===" in logs and "===OUTPUT_END===" in logs:
            start = logs.find("===OUTPUT_START===") + len("===OUTPUT_START===")
            end = logs.find("===OUTPUT_END===")
            output = logs[start:end].strip()
        else:
            # No markers found, use all as output
            output = logs.strip()

        # Extract error between markers
        if "===ERROR_START===" in logs and "===ERROR_END===" in logs:
            start = logs.find("===ERROR_START===") + len("===ERROR_START===")
            end = logs.find("===ERROR_END===")
            error_content = logs[start:end].strip()
            if error_content:
                error = error_content

        # If exit code is non-zero and no error captured, use generic message
        if exit_code != 0 and not error:
            error = f"Container exited with code {exit_code}"

        return output, error

    async def get_resource_usage(self) -> ResourceUsageSummary:
        """Get resource usage from last execution.

        Returns:
            ResourceUsageSummary with metrics
        """
        return await self._monitor.stop_monitoring()

    def is_docker_available(self) -> bool:
        """Check if Docker is available and accessible.

        Returns:
            True if Docker is available
        """
        is_installed, _ = SecurityValidator.validate_docker_installation()
        return is_installed
