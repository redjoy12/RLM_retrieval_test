"""Sandbox factory for creating appropriate sandbox instances."""

import os
from typing import Optional

import structlog

from rlm.config import get_docker_settings, get_settings
from rlm.sandbox.base import REPLSandboxInterface
from rlm.sandbox.docker_repl import DockerREPLSandbox
from rlm.sandbox.local_repl import LocalREPLSandbox

logger = structlog.get_logger()


class SandboxFactory:
    """Factory for creating REPL sandbox instances.

    Provides automatic sandbox selection based on environment and configuration.
    Supports manual selection of specific sandbox types.

    Example:
        ```python
        # Auto-select based on environment
        sandbox = SandboxFactory.create_sandbox("auto")

        # Force specific sandbox type
        docker_sandbox = SandboxFactory.create_sandbox("docker")
        local_sandbox = SandboxFactory.create_sandbox("local")

        # With custom configuration
        sandbox = SandboxFactory.create_sandbox(
            "docker",
            memory_limit="1g",
            security_profile="strict"
        )
        ```
    """

    @staticmethod
    def create_sandbox(
        sandbox_type: str = "auto",
        **kwargs,
    ) -> REPLSandboxInterface:
        """Create a sandbox instance of the specified type.

        Args:
            sandbox_type: Type of sandbox to create:
                - "auto": Automatically select based on environment
                - "local": LocalREPLSandbox (fast, less secure)
                - "docker": DockerREPLSandbox (secure, isolated)
            **kwargs: Additional arguments passed to sandbox constructor

        Returns:
            REPLSandboxInterface implementation

        Raises:
            ValueError: If sandbox_type is unknown
            RuntimeError: If required dependencies not available
        """
        if sandbox_type == "auto":
            sandbox_type = SandboxFactory._auto_select_sandbox()
            logger.info("sandbox_auto_selected", selected_type=sandbox_type)

        if sandbox_type == "local":
            return SandboxFactory._create_local_sandbox(**kwargs)

        elif sandbox_type == "docker":
            return SandboxFactory._create_docker_sandbox(**kwargs)

        else:
            raise ValueError(
                f"Unknown sandbox type: {sandbox_type}. "
                f"Available: auto, local, docker"
            )

    @staticmethod
    def _auto_select_sandbox() -> str:
        """Automatically select appropriate sandbox type.

        Selection logic:
        1. Check environment variable RLM_SANDBOX_TYPE
        2. Check if running in production mode
        3. Check if Docker is available
        4. Default to local

        Returns:
            Sandbox type string
        """
        # Check explicit environment variable
        env_type = os.getenv("RLM_SANDBOX_TYPE", "").lower()
        if env_type in ("local", "docker"):
            return env_type

        # Check production mode
        if os.getenv("RLM_ENVIRONMENT", "").lower() == "production":
            # In production, prefer Docker if available
            if SandboxFactory._is_docker_available():
                return "docker"
            logger.warning(
                "production_mode_no_docker",
                message="Running in production but Docker not available",
            )
            return "local"

        # Check if Docker is available for development
        if SandboxFactory._is_docker_available():
            return "docker"

        # Default to local
        return "local"

    @staticmethod
    def _is_docker_available() -> bool:
        """Check if Docker is installed and accessible.

        Returns:
            True if Docker is available
        """
        try:
            import docker

            client = docker.from_env()
            client.ping()
            return True
        except Exception:
            return False

    @staticmethod
    def _create_local_sandbox(**kwargs) -> LocalREPLSandbox:
        """Create a LocalREPLSandbox instance.

        Args:
            **kwargs: Arguments for LocalREPLSandbox

        Returns:
            LocalREPLSandbox instance
        """
        settings = get_settings()

        # Apply default settings if not provided
        if "timeout" not in kwargs:
            kwargs["timeout"] = settings.code_execution_timeout

        logger.debug("creating_local_sandbox", kwargs=kwargs)
        return LocalREPLSandbox(**kwargs)

    @staticmethod
    def _create_docker_sandbox(**kwargs) -> DockerREPLSandbox:
        """Create a DockerREPLSandbox instance.

        Args:
            **kwargs: Arguments for DockerREPLSandbox

        Returns:
            DockerREPLSandbox instance

        Raises:
            RuntimeError: If Docker not available
        """
        # Validate Docker is available
        if not SandboxFactory._is_docker_available():
            raise RuntimeError(
                "Docker not available. Please install Docker and ensure "
                "the daemon is running, or use sandbox_type='local'"
            )

        settings = get_settings()
        docker_settings = get_docker_settings()

        # Apply default settings if not provided
        if "image" not in kwargs:
            kwargs["image"] = docker_settings.image
        if "timeout" not in kwargs:
            kwargs["timeout"] = settings.code_execution_timeout
        if "memory_limit" not in kwargs:
            kwargs["memory_limit"] = docker_settings.memory_limit
        if "cpu_limit" not in kwargs:
            kwargs["cpu_limit"] = docker_settings.cpu_limit
        if "network_enabled" not in kwargs:
            kwargs["network_enabled"] = docker_settings.network_enabled
        if "security_profile" not in kwargs:
            kwargs["security_profile"] = docker_settings.security_profile
        if "auto_cleanup" not in kwargs:
            kwargs["auto_cleanup"] = docker_settings.auto_cleanup

        logger.debug("creating_docker_sandbox", kwargs=kwargs)
        return DockerREPLSandbox(**kwargs)

    @staticmethod
    def get_available_types() -> list[str]:
        """Get list of available sandbox types.

        Returns:
            List of available sandbox type names
        """
        types = ["auto", "local"]

        if SandboxFactory._is_docker_available():
            types.append("docker")

        return types

    @staticmethod
    def get_recommended_type() -> str:
        """Get recommended sandbox type for current environment.

        Returns:
            Recommended sandbox type name
        """
        return SandboxFactory._auto_select_sandbox()


# Convenience function for easy imports
def create_sandbox(
    sandbox_type: str = "auto",
    **kwargs,
) -> REPLSandboxInterface:
    """Create a sandbox instance (convenience function).

    This is a shorthand for SandboxFactory.create_sandbox().

    Args:
        sandbox_type: Type of sandbox (auto, local, docker)
        **kwargs: Additional sandbox configuration

    Returns:
        REPLSandboxInterface implementation
    """
    return SandboxFactory.create_sandbox(sandbox_type, **kwargs)
