"""Custom exceptions for RLM."""


class RLMError(Exception):
    """Base exception for RLM errors."""
    pass


class RecursionLimitError(RLMError):
    """Raised when recursion depth limit is reached."""
    pass


class CodeExecutionError(RLMError):
    """Raised when code execution fails."""
    
    def __init__(self, message: str, code: str = "", output: str = ""):
        super().__init__(message)
        self.code = code
        self.output = output


class TimeoutError(RLMError):
    """Raised when execution times out."""
    pass


class LLMError(RLMError):
    """Raised when LLM call fails."""
    
    def __init__(self, message: str, provider: str = "", model: str = ""):
        super().__init__(message)
        self.provider = provider
        self.model = model


class SandboxError(RLMError):
    """Raised when sandbox operation fails."""
    pass


class ConfigurationError(RLMError):
    """Raised when configuration is invalid."""
    pass
