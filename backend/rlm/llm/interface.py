"""LLM Client interface and implementations using LiteLLM."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Optional

from rlm.types import LLMResponse


class LLMClientInterface(ABC):
    """Abstract interface for LLM clients.
    
    This interface abstracts away the specific LLM provider,
    allowing the RLM engine to work with any provider.
    """
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion from the LLM.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0 - 2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse with content and metadata
        """
        ...
    
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming completion from the LLM.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Yields:
            Chunks of the generated text
        """
        ...
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the model being used.
        
        Returns:
            Model name string
        """
        ...
