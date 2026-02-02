"""Example: Using Docker Sandbox for Secure Code Execution.

This example demonstrates how to use the Docker-based REPL sandbox
for secure, isolated code execution.

Requirements:
    - Docker Engine installed and running
    - docker Python SDK installed (pip install docker)

Usage:
    python examples/docker_sandbox_usage.py
"""

import asyncio
import os

from rlm import DockerREPLSandbox, SandboxFactory, create_sandbox
from rlm.sandbox.security import SecurityProfiles


async def example_basic_docker_execution():
    """Example 1: Basic Docker sandbox execution."""
    print("\n=== Example 1: Basic Docker Execution ===\n")

    # Check if Docker is available
    sandbox = DockerREPLSandbox()
    if not sandbox.is_docker_available():
        print("Docker not available. Please install Docker and ensure the daemon is running.")
        return

    print("Docker is available!")

    # Simple callback for sub-LLM calls
    def mock_llm_callback(query: str, chunk: str | None) -> str:
        return f"[Mock response for: {query[:50]}...]"

    # Execute code in Docker container
    result = await sandbox.execute(
        code="""
import math
print(f"Pi = {math.pi}")
print(f"Square root of 16 = {math.sqrt(16)}")
result = "Hello from Docker!"
print(result)
""",
        context=None,
        sub_llm_callback=mock_llm_callback,
        timeout=30,
    )

    print(f"Output:\n{result.output}")
    print(f"Error: {result.error}")
    print(f"Execution time: {result.execution_time_ms:.2f}ms")
    print(f"Memory usage: {result.memory_usage_mb:.2f}MB")


async def example_with_security_profiles():
    """Example 2: Using different security profiles."""
    print("\n=== Example 2: Security Profiles ===\n")

    def mock_callback(query: str, chunk: str | None) -> str:
        return "mock response"

    # Standard profile (balanced security)
    print("Using STANDARD security profile...")
    sandbox_standard = DockerREPLSandbox(
        memory_limit="256m",
        security_profile="standard",
    )

    result = await sandbox_standard.execute(
        code="print('Running with standard security')",
        context=None,
        sub_llm_callback=mock_callback,
        timeout=10,
    )
    print(f"Standard: {result.output.strip()}")

    # Development profile (relaxed security, higher limits)
    print("\nUsing DEVELOPMENT security profile...")
    sandbox_dev = DockerREPLSandbox(
        memory_limit="512m",
        cpu_limit=2.0,
        security_profile="development",
    )

    result = await sandbox_dev.execute(
        code="print('Running with development security')",
        context=None,
        sub_llm_callback=mock_callback,
        timeout=10,
    )
    print(f"Development: {result.output.strip()}")


async def example_with_context():
    """Example 3: Executing with document context."""
    print("\n=== Example 3: With Document Context ===\n")

    def mock_callback(query: str, chunk: str | None) -> str:
        return f"Analyzed chunk: {chunk[:50] if chunk else 'None'}..."

    # Create sample document context
    document = """
The quick brown fox jumps over the lazy dog.
Python is a powerful programming language.
Docker provides containerization for applications.
"""

    sandbox = DockerREPLSandbox(memory_limit="256m")

    result = await sandbox.execute(
        code="""
# Access the context variable
lines = context.strip().split('\\n')
print(f"Document has {len(lines)} lines")
print(f"First line: {lines[0] if lines else 'Empty'}")
""",
        context=document,
        sub_llm_callback=mock_callback,
        timeout=10,
    )

    print(f"Output:\n{result.output}")


async def example_factory_pattern():
    """Example 4: Using the factory pattern."""
    print("\n=== Example 4: Factory Pattern ===\n")

    def mock_callback(query: str, chunk: str | None) -> str:
        return "factory response"

    # Check available sandbox types
    available = SandboxFactory.get_available_types()
    print(f"Available sandbox types: {available}")

    recommended = SandboxFactory.get_recommended_type()
    print(f"Recommended type: {recommended}")

    # Create sandbox using factory
    print("\nCreating sandbox using factory (auto mode)...")
    sandbox = create_sandbox("auto")
    print(f"Created: {sandbox.get_sandbox_type()}")

    result = await sandbox.execute(
        code="print('Factory-created sandbox works!')",
        context=None,
        sub_llm_callback=mock_callback,
        timeout=10,
    )
    print(f"Output: {result.output.strip()}")


async def example_resource_monitoring():
    """Example 5: Resource monitoring."""
    print("\n=== Example 5: Resource Monitoring ===\n")

    def mock_callback(query: str, chunk: str | None) -> str:
        return "monitoring"

    sandbox = DockerREPLSandbox(memory_limit="256m")

    # Execute some code
    await sandbox.execute(
        code="""
import math
# Do some computation
for i in range(1000):
    x = math.sqrt(i)
print("Computation complete!")
""",
        context=None,
        sub_llm_callback=mock_callback,
        timeout=30,
    )

    # Get resource usage summary
    summary = await sandbox.get_resource_usage()

    print(f"Execution time: {summary.execution_time_ms:.2f}ms")
    print(f"Peak memory: {summary.peak_memory_mb:.2f}MB")
    print(f"Peak CPU: {summary.peak_cpu_percent:.2f}%")
    print(f"Average memory: {summary.average_memory_mb:.2f}MB")


async def main():
    """Run all examples."""
    print("RLM Document Retrieval - Docker Sandbox Examples")
    print("=" * 50)

    # Check if we should run Docker examples
    if not os.getenv("RUN_DOCKER_TESTS"):
        print("\nNOTE: Docker examples are disabled by default.")
        print("Set RUN_DOCKER_TESTS=1 to run Docker integration tests.")
        print("\nThe examples below will check Docker availability...")

    try:
        await example_basic_docker_execution()
    except Exception as e:
        print(f"Example 1 failed: {e}")

    try:
        await example_with_security_profiles()
    except Exception as e:
        print(f"Example 2 failed: {e}")

    try:
        await example_with_context()
    except Exception as e:
        print(f"Example 3 failed: {e}")

    try:
        await example_factory_pattern()
    except Exception as e:
        print(f"Example 4 failed: {e}")

    try:
        await example_resource_monitoring()
    except Exception as e:
        print(f"Example 5 failed: {e}")

    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
