"""Sandbox module for code execution."""

from rlm.sandbox.base import REPLSandboxInterface
from rlm.sandbox.docker_repl import DockerREPLSandbox
from rlm.sandbox.factory import SandboxFactory, create_sandbox
from rlm.sandbox.local_repl import LocalREPLSandbox
from rlm.sandbox.monitor import ResourceMetrics, ResourceMonitor, ResourceUsageSummary
from rlm.sandbox.security import SecurityProfile, SecurityProfiles, SecurityValidator
from rlm.sandbox.utils import get_default_allowed_modules, get_default_blocked_builtins

__all__ = [
    # Base classes
    "REPLSandboxInterface",
    # Sandbox implementations
    "LocalREPLSandbox",
    "DockerREPLSandbox",
    # Factory
    "SandboxFactory",
    "create_sandbox",
    # Monitoring
    "ResourceMonitor",
    "ResourceMetrics",
    "ResourceUsageSummary",
    # Security
    "SecurityProfile",
    "SecurityProfiles",
    "SecurityValidator",
    # Utilities
    "get_default_allowed_modules",
    "get_default_blocked_builtins",
]
