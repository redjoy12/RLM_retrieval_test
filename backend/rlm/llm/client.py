"""LLM Client implementation using LiteLLM."""

import asyncio
from typing import Any, AsyncIterator, Dict, Optional

import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rlm.config import get_settings
from rlm.llm.interface import LLMClientInterface
from rlm.types import LLMResponse

logger = structlog.get_logger()


class LiteLLMClient(LLMClientInterface):
    """LLM client using LiteLLM for unified provider support.
    
    Supports 100+ providers including:
    - OpenAI (GPT-4, GPT-5, etc.)
    - Anthropic (Claude)
    - Azure OpenAI
    - Google (Gemini, Vertex AI)
    - Local models (vLLM, Ollama, etc.)
    
    Configuration is done via environment variables:
    - OPENAI_API_KEY
    - ANTHROPIC_API_KEY
    - Or set LITELLM_API_KEY for custom providers
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        """Initialize the LiteLLM client.
        
        Args:
            model: Model name (e.g., 'gpt-5-mini', 'claude-sonnet-4.5')
            provider: Provider name (e.g., 'openai', 'anthropic')
            api_base: Custom API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        settings = get_settings()
        
        self.model = model or settings.default_model
        self.provider = provider or settings.litellm_provider
        self.api_base = api_base or settings.litellm_api_base
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Build full model string for LiteLLM
        self._full_model = self._build_model_string()
        
        logger.info(
            "litellm_client_initialized",
            model=self.model,
            provider=self.provider,
            full_model=self._full_model,
        )
    
    def _build_model_string(self) -> str:
        """Build the full model string for LiteLLM.
        
        LiteLLM uses format: "provider/model" or just "model" for OpenAI
        """
        # If model already contains a slash, assume it's already formatted
        if "/" in self.model:
            return self.model
        
        # For OpenAI, we can use just the model name
        if self.provider == "openai":
            return self.model
        
        # For other providers, use provider/model format
        return f"{self.provider}/{self.model}"
    
    @retry(
        retry=retry_if_exception_type((Exception)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion using LiteLLM."""
        try:
            import litellm
            from litellm import acompletion
            
            # Configure LiteLLM
            litellm.set_verbose = False
            if self.api_base:
                litellm.api_base = self.api_base
            
            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Call LiteLLM
            response = await acompletion(
                model=self._full_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.timeout,
                **kwargs,
            )
            
            # Extract content
            content = response.choices[0].message.content
            
            # Extract usage if available
            usage = {}
            if hasattr(response, 'usage'):
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            
            logger.debug(
                "llm_generation_complete",
                model=self._full_model,
                tokens_used=usage.get("total_tokens", 0),
            )
            
            return LLMResponse(
                content=content,
                model=self._full_model,
                usage=usage,
                finish_reason=response.choices[0].finish_reason,
            )
            
        except Exception as e:
            logger.error(
                "llm_generation_failed",
                model=self._full_model,
                error=str(e),
            )
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming completion using LiteLLM."""
        try:
            import litellm
            from litellm import acompletion
            
            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Stream the response
            response = await acompletion(
                model=self._full_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.timeout,
                stream=True,
                **kwargs,
            )
            
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
                        
        except Exception as e:
            logger.error(
                "llm_streaming_failed",
                model=self._full_model,
                error=str(e),
            )
            raise
    
    def get_model_name(self) -> str:
        """Get the model name."""
        return self._full_model


class MockLLMClient(LLMClientInterface):
    """Mock LLM client for testing.
    
    Returns predefined responses or echoes the prompt.
    """
    
    def __init__(
        self,
        response_template: str = "Mock response for: {prompt}",
        delay: float = 0.1,
    ) -> None:
        """Initialize mock client.
        
        Args:
            response_template: Template for responses (can use {prompt})
            delay: Artificial delay in seconds to simulate network
        """
        self.response_template = response_template
        self.delay = delay
        self.call_count = 0
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a mock completion."""
        self.call_count += 1
        await asyncio.sleep(self.delay)
        
        content = self.response_template.format(prompt=prompt[:100])
        
        return LLMResponse(
            content=content,
            model="mock-model",
            usage={"prompt_tokens": len(prompt), "completion_tokens": len(content)},
        )
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a mock streaming completion."""
        self.call_count += 1
        content = self.response_template.format(prompt=prompt[:100])
        
        # Stream word by word
        words = content.split()
        for word in words:
            await asyncio.sleep(self.delay / len(words))
            yield word + " "
    
    def get_model_name(self) -> str:
        """Get the model name."""
        return "mock-model"
