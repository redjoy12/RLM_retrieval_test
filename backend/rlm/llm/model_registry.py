"""Model registry for managing LLM configurations and pricing."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelInfo:
    """Information about a specific LLM model."""

    provider: str
    model_id: str
    context_window: int
    supports_streaming: bool = True
    supports_functions: bool = False
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0
    api_base: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """Registry of supported LLM models with configurations and pricing."""

    def __init__(self) -> None:
        """Initialize the model registry with predefined models."""
        self._models: Dict[str, ModelInfo] = {}
        self._fallback_chains: Dict[str, List[str]] = {}
        self._register_default_models()

    def _register_default_models(self) -> None:
        """Register default models for OpenAI, Anthropic, and Local."""
        # OpenAI Models
        self.register_model(
            ModelInfo(
                provider="openai",
                model_id="gpt-4o",
                context_window=128000,
                supports_streaming=True,
                supports_functions=True,
                input_cost_per_1k=0.0025,
                output_cost_per_1k=0.010,
                capabilities=["vision", "functions", "json_mode"],
            )
        )

        self.register_model(
            ModelInfo(
                provider="openai",
                model_id="gpt-4o-mini",
                context_window=128000,
                supports_streaming=True,
                supports_functions=True,
                input_cost_per_1k=0.00015,
                output_cost_per_1k=0.0006,
                capabilities=["vision", "functions", "json_mode"],
            )
        )

        self.register_model(
            ModelInfo(
                provider="openai",
                model_id="o1",
                context_window=128000,
                supports_streaming=False,
                supports_functions=False,
                input_cost_per_1k=0.015,
                output_cost_per_1k=0.060,
                capabilities=["reasoning"],
            )
        )

        self.register_model(
            ModelInfo(
                provider="openai",
                model_id="o3-mini",
                context_window=128000,
                supports_streaming=False,
                supports_functions=False,
                input_cost_per_1k=0.0011,
                output_cost_per_1k=0.0044,
                capabilities=["reasoning"],
            )
        )

        # Anthropic Models
        self.register_model(
            ModelInfo(
                provider="anthropic",
                model_id="claude-3-5-sonnet",
                context_window=200000,
                supports_streaming=True,
                supports_functions=True,
                input_cost_per_1k=0.003,
                output_cost_per_1k=0.015,
                capabilities=["vision", "functions", "tool_use"],
            )
        )

        self.register_model(
            ModelInfo(
                provider="anthropic",
                model_id="claude-3-opus",
                context_window=200000,
                supports_streaming=True,
                supports_functions=True,
                input_cost_per_1k=0.015,
                output_cost_per_1k=0.075,
                capabilities=["vision", "functions", "tool_use", "highest_quality"],
            )
        )

        # Local Models (Ollama)
        self.register_model(
            ModelInfo(
                provider="ollama",
                model_id="llama3.2",
                context_window=128000,
                supports_streaming=True,
                supports_functions=False,
                input_cost_per_1k=0.0,
                output_cost_per_1k=0.0,
                api_base="http://localhost:11434",
                capabilities=["local", "privacy"],
            )
        )

        self.register_model(
            ModelInfo(
                provider="ollama",
                model_id="mistral",
                context_window=32768,
                supports_streaming=True,
                supports_functions=False,
                input_cost_per_1k=0.0,
                output_cost_per_1k=0.0,
                api_base="http://localhost:11434",
                capabilities=["local", "privacy"],
            )
        )

        # Set up fallback chains
        self._fallback_chains = {
            "smart": ["claude-3-opus", "gpt-4o", "claude-3-5-sonnet", "gpt-4o-mini"],
            "fast": ["gpt-4o-mini", "claude-3-5-sonnet", "gpt-4o"],
            "cheap": ["gpt-4o-mini", "ollama/llama3.2"],
            "local": ["ollama/llama3.2", "ollama/mistral"],
        }

    def register_model(self, model_info: ModelInfo) -> None:
        """Register a new model in the registry.

        Args:
            model_info: Model information to register
        """
        full_id = f"{model_info.provider}/{model_info.model_id}"
        self._models[full_id] = model_info

        # Also register without provider prefix for convenience
        if model_info.model_id not in self._models:
            self._models[model_info.model_id] = model_info

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get model information by ID.

        Args:
            model_id: Model identifier (e.g., "gpt-4o" or "openai/gpt-4o")

        Returns:
            ModelInfo if found, None otherwise
        """
        # Try exact match first
        if model_id in self._models:
            return self._models[model_id]

        # Try with common provider prefixes
        for prefix in ["openai/", "anthropic/", "ollama/"]:
            full_id = f"{prefix}{model_id}"
            if full_id in self._models:
                return self._models[full_id]

        return None

    def list_models(
        self, provider: Optional[str] = None, capability: Optional[str] = None
    ) -> List[str]:
        """List available models with optional filtering.

        Args:
            provider: Filter by provider (openai, anthropic, ollama)
            capability: Filter by capability

        Returns:
            List of model IDs
        """
        models = []
        for model_id, info in self._models.items():
            # Skip duplicate entries (with provider prefix)
            if "/" in model_id:
                continue

            if provider and info.provider != provider:
                continue

            if capability and capability not in info.capabilities:
                continue

            models.append(model_id)

        return models

    def get_fallback_chain(self, strategy: str = "smart") -> List[str]:
        """Get a fallback chain for a given strategy.

        Args:
            strategy: Fallback strategy ("smart", "fast", "cheap", "local")

        Returns:
            List of model IDs in fallback order
        """
        return self._fallback_chains.get(strategy, self._fallback_chains["smart"])

    def calculate_cost(
        self, model_id: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate the cost for a request.

        Args:
            model_id: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Total cost in USD
        """
        model = self.get_model(model_id)
        if not model:
            return 0.0

        input_cost = (input_tokens / 1000) * model.input_cost_per_1k
        output_cost = (output_tokens / 1000) * model.output_cost_per_1k

        return input_cost + output_cost

    def get_provider_for_model(self, model_id: str) -> Optional[str]:
        """Get the provider for a given model.

        Args:
            model_id: Model identifier

        Returns:
            Provider name or None
        """
        model = self.get_model(model_id)
        return model.provider if model else None

    def get_api_base(self, model_id: str) -> Optional[str]:
        """Get the API base URL for a model (for local models).

        Args:
            model_id: Model identifier

        Returns:
            API base URL or None
        """
        model = self.get_model(model_id)
        return model.api_base if model else None

    def supports_streaming(self, model_id: str) -> bool:
        """Check if a model supports streaming.

        Args:
            model_id: Model identifier

        Returns:
            True if streaming is supported
        """
        model = self.get_model(model_id)
        return model.supports_streaming if model else False

    def get_context_window(self, model_id: str) -> int:
        """Get the context window size for a model.

        Args:
            model_id: Model identifier

        Returns:
            Context window size in tokens
        """
        model = self.get_model(model_id)
        return model.context_window if model else 4096


# Global registry instance
_global_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance.

    Returns:
        ModelRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry()
    return _global_registry
