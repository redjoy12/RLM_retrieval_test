"""Cost estimation for LLM queries using tiktoken and model pricing."""

from typing import Dict, Optional, Tuple

import structlog
import tiktoken

from rlm.config import get_settings
from rlm.routing.models import CostEstimate

logger = structlog.get_logger()


class CostEstimator:
    """Estimates token counts and costs for LLM queries.
    
    Uses tiktoken for accurate OpenAI token counting and provides
    estimates for other models using character-based ratios.
    
    Example:
        ```python
        estimator = CostEstimator()
        estimate = estimator.estimate_cost(
            query="What is AI?",
            context="AI is artificial intelligence...",
            model="gpt-5-mini"
        )
        print(f"Estimated cost: ${estimate.estimated_cost_usd:.4f}")
        ```
    """
    
    # Characters per token ratio for non-OpenAI models
    CHARS_PER_TOKEN = 4
    
    # Model pricing (USD per 1K tokens) - approximate
    MODEL_PRICING: Dict[str, Dict[str, float]] = {
        # OpenAI models
        "gpt-5-nano": {"input": 0.0001, "output": 0.0004},
        "gpt-5-mini": {"input": 0.0003, "output": 0.0012},
        "gpt-5": {"input": 0.0010, "output": 0.0040},
        "gpt-4o": {"input": 0.0050, "output": 0.0150},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        
        # Anthropic models
        "claude-sonnet-4.5": {"input": 0.0030, "output": 0.0150},
        "claude-opus-4.5": {"input": 0.0150, "output": 0.0750},
        "claude-haiku": {"input": 0.00025, "output": 0.00125},
        
        # Default fallback
        "default": {"input": 0.0010, "output": 0.0030},
    }
    
    def __init__(self, cost_buffer_percent: int = 10) -> None:
        """Initialize the cost estimator.
        
        Args:
            cost_buffer_percent: Safety buffer percentage to add to estimates
        """
        self.cost_buffer_percent = cost_buffer_percent
        self._encoders: Dict[str, tiktoken.Encoding] = {}
        
        logger.info(
            "cost_estimator_initialized",
            cost_buffer=cost_buffer_percent,
        )
    
    def estimate_cost(
        self,
        query: str,
        context: str,
        model: Optional[str] = None,
        expected_output_tokens: Optional[int] = None,
    ) -> CostEstimate:
        """Estimate cost for a query with context.
        
        Args:
            query: The user's query
            context: Document context to include
            model: Model name (defaults to settings)
            expected_output_tokens: Expected output length (auto-estimated if None)
            
        Returns:
            CostEstimate with detailed breakdown
        """
        settings = get_settings()
        model = model or settings.default_model
        
        # Count input tokens
        input_tokens = self._count_tokens(query + "\n\n" + context, model)
        
        # Estimate output tokens if not provided
        if expected_output_tokens is None:
            output_tokens = self._estimate_output_tokens(query, model)
        else:
            output_tokens = expected_output_tokens
        
        # Calculate cost
        cost = self._calculate_cost(input_tokens, output_tokens, model)
        
        estimate = CostEstimate(
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
            estimated_total_tokens=input_tokens + output_tokens,
            estimated_cost_usd=cost,
            model_used=model,
            cost_buffer_percent=self.cost_buffer_percent,
        )
        
        logger.debug(
            "cost_estimated",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
        
        return estimate
    
    def estimate_rag_cost(
        self,
        query: str,
        num_chunks: int,
        avg_chunk_size: int = 5000,  # chars per chunk
        model: Optional[str] = None,
    ) -> CostEstimate:
        """Estimate cost for RAG retrieval + answer.
        
        Args:
            query: The user's query
            num_chunks: Number of chunks to retrieve
            avg_chunk_size: Average chunk size in characters
            model: Model name
            
        Returns:
            CostEstimate for RAG approach
        """
        # Estimate context from retrieved chunks
        estimated_context_chars = num_chunks * avg_chunk_size
        estimated_context = "x" * estimated_context_chars  # Dummy content
        
        # RAG typically has shorter outputs
        return self.estimate_cost(
            query=query,
            context=estimated_context,
            model=model,
            expected_output_tokens=500,  # RAG answers are typically shorter
        )
    
    def estimate_rlm_cost(
        self,
        query: str,
        context_size: int,
        estimated_sub_llm_calls: int,
        model: Optional[str] = None,
    ) -> CostEstimate:
        """Estimate cost for RLM recursive analysis.
        
        Args:
            query: The user's query
            context_size: Total context size in characters
            estimated_sub_llm_calls: Expected number of sub-LLM calls
            model: Model name
            
        Returns:
            CostEstimate for RLM approach
        """
        # Base cost for root LLM
        dummy_context = "x" * min(context_size, 10000)  # Root sees limited context
        base_estimate = self.estimate_cost(query, dummy_context, model)
        
        # Cost per sub-LLM call (smaller context per call)
        sub_call_context = "x" * 2000  # ~500 tokens per chunk
        sub_call_estimate = self.estimate_cost("sub-query", sub_call_context, model)
        
        # Total cost = root + (sub_calls * cost_per_call)
        total_input = base_estimate.estimated_input_tokens + \
                     (estimated_sub_llm_calls * sub_call_estimate.estimated_input_tokens)
        total_output = base_estimate.estimated_output_tokens + \
                      (estimated_sub_llm_calls * sub_call_estimate.estimated_output_tokens)
        
        # Add code generation overhead
        code_gen_tokens = 1000  # For generated Python code
        total_output += code_gen_tokens
        
        total_cost = self._calculate_cost(total_input, total_output, model)
        
        return CostEstimate(
            estimated_input_tokens=total_input,
            estimated_output_tokens=total_output,
            estimated_total_tokens=total_input + total_output,
            estimated_cost_usd=total_cost,
            model_used=model,
            cost_buffer_percent=self.cost_buffer_percent,
        )
    
    def _count_tokens(self, text: str, model: str) -> int:
        """Count tokens in text for a given model."""
        # Try to use tiktoken for OpenAI models
        if "gpt" in model.lower() or "openai" in model.lower():
            return self._count_tokens_tiktoken(text, model)
        
        # Fall back to character-based estimation
        return len(text) // self.CHARS_PER_TOKEN
    
    def _count_tokens_tiktoken(self, text: str, model: str) -> int:
        """Count tokens using tiktoken for OpenAI models."""
        try:
            # Map model names to tiktoken encodings
            encoder_name = self._get_encoder_name(model)
            
            if encoder_name not in self._encoders:
                self._encoders[encoder_name] = tiktoken.get_encoding(encoder_name)
            
            encoder = self._encoders[encoder_name]
            tokens = encoder.encode(text)
            return len(tokens)
            
        except Exception as e:
            logger.warning(
                "tiktoken_counting_failed",
                model=model,
                error=str(e),
                fallback="character_based",
            )
            # Fall back to character-based
            return len(text) // self.CHARS_PER_TOKEN
    
    def _get_encoder_name(self, model: str) -> str:
        """Get tiktoken encoder name for a model."""
        # Map common model names to encoders
        model_lower = model.lower()
        
        if "gpt-5" in model_lower:
            return "o200k_base"  # GPT-5 uses o200k_base
        elif "gpt-4o" in model_lower:
            return "o200k_base"
        elif "gpt-4" in model_lower:
            return "cl100k_base"
        elif "gpt-3.5" in model_lower:
            return "cl100k_base"
        else:
            return "cl100k_base"  # Default fallback
    
    def _estimate_output_tokens(self, query: str, model: str) -> int:
        """Estimate output tokens based on query complexity."""
        query_lower = query.lower()
        
        # Base output sizes
        if any(kw in query_lower for kw in ["list", "enumerate", "what are"]):
            return 800  # List responses
        elif any(kw in query_lower for kw in ["summarize", "summary", "brief"]):
            return 600  # Summaries
        elif any(kw in query_lower for kw in ["explain", "describe", "how does"]):
            return 1200  # Explanations
        elif any(kw in query_lower for kw in ["compare", "contrast", "difference"]):
            return 1500  # Comparisons
        elif any(kw in query_lower for kw in ["analyze", "evaluation", "assessment"]):
            return 2000  # Analysis
        else:
            return 1000  # Default
    
    def _calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
    ) -> float:
        """Calculate cost in USD."""
        # Get pricing for model
        pricing = self._get_model_pricing(model)
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    def _get_model_pricing(self, model: str) -> Dict[str, float]:
        """Get pricing for a model."""
        model_lower = model.lower()
        
        # Try exact match
        if model_lower in self.MODEL_PRICING:
            return self.MODEL_PRICING[model_lower]
        
        # Try partial match
        for model_name, pricing in self.MODEL_PRICING.items():
            if model_name in model_lower:
                return pricing
        
        # Return default
        return self.MODEL_PRICING["default"]
    
    def compare_strategies(
        self,
        query: str,
        context_size: int,
        model: Optional[str] = None,
    ) -> Dict[str, CostEstimate]:
        """Compare costs across all strategies.
        
        Returns:
            Dictionary mapping strategy names to cost estimates
        """
        dummy_context = "x" * context_size
        
        # Direct LLM (whole context)
        direct_cost = self.estimate_cost(query, dummy_context, model)
        
        # RAG (top 5 chunks)
        rag_cost = self.estimate_rag_cost(query, num_chunks=5, model=model)
        
        # RLM (with estimated sub-calls)
        estimated_calls = min(context_size // 5000, 20)  # Heuristic
        rlm_cost = self.estimate_rlm_cost(
            query, context_size, estimated_calls, model
        )
        
        # Hybrid (RAG + RLM on filtered)
        hybrid_cost = self.estimate_rlm_cost(
            query, context_size // 5, estimated_calls // 2, model
        )
        
        return {
            "direct_llm": direct_cost,
            "rag": rag_cost,
            "rlm": rlm_cost,
            "hybrid": hybrid_cost,
        }
