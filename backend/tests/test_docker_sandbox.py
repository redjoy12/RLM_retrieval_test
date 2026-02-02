"""Tests for Docker REPL sandbox implementation."""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rlm.sandbox.docker_repl import DockerREPLSandbox
from rlm.sandbox.factory import SandboxFactory, create_sandbox
from rlm.sandbox.security import SecurityProfile, SecurityProfiles, SecurityValidator


# Mark tests that require Docker
pytestmark = [
    pytest.mark.asyncio,
]


class TestSecurityProfiles:
    """Test security profile configurations."""

    def test_strict_profile(self):
        """Test strict security profile."""
        profile = SecurityProfiles.strict()

        assert profile.name == "strict"
        assert profile.read_only is True
        assert "ALL" in profile.cap_drop
        assert profile.network_mode == "none"
        assert profile.pids_limit == 50

    def test_standard_profile(self):
        """Test standard security profile."""
        profile = SecurityProfiles.standard()

        assert profile.name == "standard"
        assert profile.read_only is True
        assert profile.network_mode == "bridge"
        assert profile.pids_limit == 100

    def test_development_profile(self):
        """Test development security profile."""
        profile = SecurityProfiles.development()

        assert profile.name == "development"
        assert profile.read_only is False
        assert profile.network_mode == "bridge"

    def test_profile_custom_limits(self):
        """Test custom resource limits in profiles."""
        profile = SecurityProfiles.strict(memory_limit="1g", cpu_limit=2.0)

        assert profile.memory_limit == "1g"
        assert profile.cpu_limit == 2.0

    def test_profile_to_docker_config(self):
        """Test converting profile to Docker config."""
        profile = SecurityProfiles.standard()
        config = profile.to_docker_config()

        assert config["ReadonlyRootfs"] is True
        assert config["NetworkMode"] == "bridge"
        assert config["PidsLimit"] == 100
        assert "Memory" in config
        assert "NanoCpus" in config

    def test_memory_parsing(self):
        """Test memory limit parsing."""
        profile = SecurityProfile(name="test", memory_limit="512m")
        bytes_val = profile._parse_memory("512m")

        assert bytes_val == 512 * 1024 * 1024

    def test_invalid_profile_name(self):
        """Test getting invalid profile name."""
        with pytest.raises(ValueError, match="Unknown security profile"):
            SecurityProfiles.get_profile("invalid")


class TestSecurityValidator:
    """Test security validation utilities."""

    def test_check_code_safety_dangerous_imports(self):
        """Test detection of dangerous imports."""
        code = "import os; os.system('ls')"
        is_safe, warnings = SecurityValidator.check_code_safety(code)

        assert is_safe is False
        assert len(warnings) > 0
        assert any("os.system" in w for w in warnings)

    def test_check_code_safety_dangerous_builtins(self):
        """Test detection of dangerous builtins."""
        code = "eval('1 + 1')"
        is_safe, warnings = SecurityValidator.check_code_safety(code)

        assert is_safe is False
        assert any("eval" in w for w in warnings)

    def test_check_code_safety_wildcard_import(self):
        """Test detection of wildcard imports."""
        code = "from os import *"
        is_safe, warnings = SecurityValidator.check_code_safety(code)

        # Wildcard is warning but not unsafe
        assert any("wildcard" in w.lower() for w in warnings)

    def test_validate_docker_installation(self):
        """Test Docker installation check."""
        # This test checks if the function runs without error
        # Result depends on system state
        is_installed, message = SecurityValidator.validate_docker_installation()

        assert isinstance(is_installed, bool)
        assert isinstance(message, str)


class TestSandboxFactory:
    """Test sandbox factory."""

    def test_get_available_types(self):
        """Test getting available sandbox types."""
        types = SandboxFactory.get_available_types()

        assert "auto" in types
        assert "local" in types

    def test_create_local_sandbox(self):
        """Test creating local sandbox."""
        sandbox = SandboxFactory.create_sandbox("local")

        assert sandbox.get_sandbox_type() == "local"

    def test_create_sandbox_auto(self):
        """Test auto-creating sandbox."""
        sandbox = SandboxFactory.create_sandbox("auto")

        # Should create either local or docker depending on environment
        assert sandbox.get_sandbox_type() in ["local", "docker"]

    def test_create_unknown_sandbox_type(self):
        """Test creating unknown sandbox type."""
        with pytest.raises(ValueError, match="Unknown sandbox type"):
            SandboxFactory.create_sandbox("unknown")

    def test_convenience_function(self):
        """Test create_sandbox convenience function."""
        sandbox = create_sandbox("local")

        assert sandbox.get_sandbox_type() == "local"


@pytest.mark.skipif(
    not os.getenv("RUN_DOCKER_TESTS"),
    reason="Docker tests disabled. Set RUN_DOCKER_TESTS=1 to enable."
)
class TestDockerSandboxIntegration:
    """Integration tests for Docker sandbox (requires Docker)."""

    async def test_docker_sandbox_basic_execution(self):
        """Test basic code execution in Docker sandbox."""
        sandbox = DockerREPLSandbox(
            memory_limit="256m",
            cpu_limit=0.5,
            auto_cleanup=True,
        )

        def mock_callback(query, chunk):
            return f"Response: {query}"

        result = await sandbox.execute(
            code="print('Hello from Docker!')",
            context=None,
            sub_llm_callback=mock_callback,
            timeout=30,
        )

        assert "Hello from Docker!" in result.output
        assert result.error is None or result.error == ""

    async def test_docker_sandbox_with_context(self):
        """Test Docker sandbox with document context."""
        sandbox = DockerREPLSandbox(memory_limit="256m")

        def mock_callback(query, chunk):
            return "mock response"

        context = "This is test document content."

        result = await sandbox.execute(
            code="print(len(context))",
            context=context,
            sub_llm_callback=mock_callback,
            timeout=30,
        )

        # Should output the length of context
        assert result.output.strip() == str(len(context))

    async def test_docker_sandbox_timeout(self):
        """Test Docker sandbox timeout enforcement."""
        sandbox = DockerREPLSandbox(memory_limit="256m")

        def mock_callback(query, chunk):
            return "mock"

        result = await sandbox.execute(
            code="import time; time.sleep(10)",
            context=None,
            sub_llm_callback=mock_callback,
            timeout=2,  # Very short timeout
        )

        assert result.error is not None
        assert "timeout" in result.error.lower()

    async def test_docker_sandbox_security_profile(self):
        """Test Docker sandbox with strict security profile."""
        sandbox = DockerREPLSandbox(
            memory_limit="256m",
            security_profile="strict",
        )

        def mock_callback(query, chunk):
            return "mock"

        # Try to access network (should fail in strict mode)
        result = await sandbox.execute(
            code="import urllib.request; print('should not work')",
            context=None,
            sub_llm_callback=mock_callback,
            timeout=10,
        )

        # Should either fail or not have network access
        assert result.error is not None or "should not work" not in result.output

    async def test_docker_resource_monitoring(self):
        """Test resource monitoring in Docker sandbox."""
        sandbox = DockerREPLSandbox(memory_limit="256m")

        def mock_callback(query, chunk):
            return "mock"

        await sandbox.execute(
            code="print('test')",
            context=None,
            sub_llm_callback=mock_callback,
            timeout=30,
        )

        # Get resource usage summary
        summary = await sandbox.get_resource_usage()

        assert summary.execution_time_ms > 0
        assert summary.peak_memory_mb >= 0

    async def test_docker_is_available(self):
        """Test Docker availability check."""
        sandbox = DockerREPLSandbox()

        # Should return boolean
        is_available = sandbox.is_docker_available()
        assert isinstance(is_available, bool)


class TestDockerSandboxMocked:
    """Unit tests for Docker sandbox with mocked Docker client."""

    @pytest.fixture
    def mock_docker_client(self):
        """Create mock Docker client."""
        with patch("docker.from_env") as mock_from_env:
            mock_client = MagicMock()
            mock_from_env.return_value = mock_client
            yield mock_client

    async def test_docker_execution_with_mock(self, mock_docker_client):
        """Test Docker execution with mocked client."""
        # Setup mock container
        mock_container = MagicMock()
        mock_container.short_id = "abc123"
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.return_value = b"===OUTPUT_START===\nHello\n===OUTPUT_END==="

        mock_docker_client.containers.create.return_value = mock_container
        mock_docker_client.images.get.side_effect = Exception("Not found")

        sandbox = DockerREPLSandbox()
        sandbox._docker_client = mock_docker_client

        def mock_callback(query, chunk):
            return "mock"

        result = await sandbox.execute(
            code="print('Hello')",
            context=None,
            sub_llm_callback=mock_callback,
        )

        assert "Hello" in result.output
        mock_container.start.assert_called_once()

    async def test_docker_not_available(self):
        """Test handling when Docker is not available."""
        with patch(
            "rlm.sandbox.security.SecurityValidator.validate_docker_installation",
            return_value=(False, "Docker not installed"),
        ):
            sandbox = DockerREPLSandbox()

            def mock_callback(query, chunk):
                return "mock"

            result = await sandbox.execute(
                code="print('test')",
                context=None,
                sub_llm_callback=mock_callback,
            )

            assert result.error is not None
            assert "Docker not available" in result.error


class TestSandboxComparison:
    """Compare behavior between local and Docker sandboxes."""

    async def test_both_sandboxes_execute_code(self):
        """Test that both sandbox types can execute code."""
        code = "x = 2 + 2; print(x)"

        def mock_callback(query, chunk):
            return "mock"

        # Local sandbox
        from rlm.sandbox.local_repl import LocalREPLSandbox
        local = LocalREPLSandbox(timeout=5)
        local_result = await local.execute(code, None, mock_callback)

        assert "4" in local_result.output

        # Docker sandbox (if available)
        docker = DockerREPLSandbox(memory_limit="256m")
        if docker.is_docker_available():
            docker_result = await docker.execute(code, None, mock_callback)
            assert "4" in docker_result.output

    async def test_sandbox_type_identification(self):
        """Test sandbox type identification."""
        from rlm.sandbox.local_repl import LocalREPLSandbox

        local = LocalREPLSandbox()
        assert local.get_sandbox_type() == "local"

        docker = DockerREPLSandbox()
        assert docker.get_sandbox_type() == "docker"
