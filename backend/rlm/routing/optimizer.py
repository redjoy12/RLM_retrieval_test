"""Query optimizer using LLM for intelligent suggestions."""

import structlog

from rlm.config import get_settings
from rlm.llm.client import LiteLLMClient
from rlm.routing.models import OptimizationResult, QueryComplexity, QuerySuggestion

logger = structlog.get_logger()


class QueryOptimizer:
    """Optimizes queries using LLM-based analysis.
    
    Provides intelligent suggestions for query reformulation,
    decomposition, and improvement.
    
    Example:
        ```python
        optimizer = QueryOptimizer()
        result = await optimizer.optimize(
            "Compare and analyze all the findings",
            complexity=QueryComplexity.QUADRATIC
        )
        print(result.suggestions[0].explanation)
        ```
    """
    
    def __init__(self, llm_client: LiteLLMClient = None) -> None:
        """Initialize query optimizer."""
        self.settings = get_settings()
        self.llm_client = llm_client or LiteLLMClient()
        
        logger.info("query_optimizer_initialized")
    
    async def optimize(
        self,
        query: str,
        complexity: QueryComplexity,
        context_size: int = 0,
    ) -> OptimizationResult:
        """Optimize a query and provide suggestions."""
        suggestions = []
        
        # Get LLM-based suggestions
        llm_suggestions = await self._get_llm_suggestions(query, complexity)
        suggestions.extend(llm_suggestions)
        
        # Add rule-based suggestions
        rule_suggestions = self._get_rule_suggestions(query, complexity, context_size)
        suggestions.extend(rule_suggestions)
        
        # Determine if we should use optimized query
        use_optimized = any(s.confidence > 0.7 for s in suggestions)
        
        # Build optimized query from best suggestions
        optimized_query = None
        if use_optimized:
            optimized_query = self._build_optimized_query(query, suggestions)
        
        result = OptimizationResult(
            original_query=query,
            suggestions=suggestions,
            optimized_query=optimized_query,
            use_optimized=use_optimized,
        )
        
        logger.debug(
            "query_optimized",
            original=query[:50],
            suggestions_count=len(suggestions),
            use_optimized=use_optimized,
        )
        
        return result
    
    async def _get_llm_suggestions(
        self,
        query: str,
        complexity: QueryComplexity,
    ) -> list:
        """Get suggestions from LLM analysis."""
        suggestions = []
        
        # Only use LLM for complex queries
        if complexity in [QueryComplexity.SIMPLE, QueryComplexity.SINGLE_HOP]:
            return suggestions
        
        try:
            prompt = f"""Analyze this query and suggest improvements:

Query: "{query}"
Complexity Level: {complexity.value}

Provide suggestions in JSON format:
{{
    "reformulation": "improved version of the query",
    "decomposition": ["step 1", "step 2", "step 3"],
    "keywords": ["keyword1", "keyword2"],
    "explanation": "why these improvements help"
}}"""
            
            response = await self.llm_client.generate(
                prompt=prompt,
                system_prompt="You are a query optimization expert.",
                temperature=0.3,
            )
            
            # Parse suggestions
            import json
            try:
                data = json.loads(response.content)
                
                if "reformulation" in data:
                    suggestions.append(QuerySuggestion(
                        suggestion_type="reformulation",
                        original_query=query,
                        suggested_query=data["reformulation"],
                        explanation=data.get("explanation", ""),
                        confidence=0.8,
                        expected_improvement="Clearer query formulation",
                    ))
                
                if "decomposition" in data and data["decomposition"]:
                    suggestions.append(QuerySuggestion(
                        suggestion_type="decomposition",
                        original_query=query,
                        explanation="Break into sub-queries: " + ", ".join(data["decomposition"]),
                        confidence=0.75,
                        expected_improvement="Better handling via sub-LLM calls",
                    ))
                
            except json.JSONDecodeError:
                logger.warning("failed_to_parse_llm_suggestions")
                
        except Exception as e:
            logger.error("llm_optimization_failed", error=str(e))
        
        return suggestions
    
    def _get_rule_suggestions(
        self,
        query: str,
        complexity: QueryComplexity,
        context_size: int,
    ) -> list:
        """Get rule-based suggestions."""
        suggestions = []
        
        # Suggest chunking strategy based on context size
        if context_size > 500000:
            suggestions.append(QuerySuggestion(
                suggestion_type="chunking",
                original_query=query,
                explanation="Large context detected. Use smaller chunk sizes (50K chars) for better retrieval.",
                confidence=0.9,
                expected_improvement="Improved retrieval accuracy",
            ))
        
        # Suggest decomposition for complex queries
        if complexity == QueryComplexity.QUADRATIC:
            suggestions.append(QuerySuggestion(
                suggestion_type="decomposition",
                original_query=query,
                explanation="Complex aggregation query. Consider breaking into: 1) Identify scope 2) Aggregate findings 3) Synthesize results",
                confidence=0.85,
                expected_improvement="Better RLM performance with focused sub-queries",
            ))
        
        return suggestions
    
    def _build_optimized_query(self, query: str, suggestions: list) -> str:
        """Build optimized query from suggestions."""
        # Prioritize reformulation
        reformulations = [s for s in suggestions if s.suggestion_type == "reformulation"]
        if reformulations:
            best = max(reformulations, key=lambda s: s.confidence)
            return best.suggested_query or query
        
        return query


class SimpleOptimizer:
    """Lightweight rule-based optimizer without LLM calls."""
    
    def optimize(self, query: str) -> OptimizationResult:
        """Simple optimization using heuristics."""
        suggestions = []
        
        # Check for vague terms
        vague_terms = ["everything", "all", "entire"]
        if any(term in query.lower() for term in vague_terms):
            suggestions.append(QuerySuggestion(
                suggestion_type="keywords",
                original_query=query,
                explanation="Consider using specific terms instead of vague words like 'everything' or 'all'",
                confidence=0.6,
            ))
        
        return OptimizationResult(
            original_query=query,
            suggestions=suggestions,
            use_optimized=False,
        )
