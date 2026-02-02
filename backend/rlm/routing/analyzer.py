"""Query complexity analyzer for routing decisions."""

import re
from typing import List, Set, Tuple

import structlog

from rlm.routing.models import QueryAnalysis, QueryComplexity

logger = structlog.get_logger()


class QueryAnalyzer:
    """Analyzes queries to determine complexity and routing needs.
    
    Uses keyword-based heuristics and pattern matching to classify
    queries into complexity categories.
    
    Example:
        ```python
        analyzer = QueryAnalyzer()
        analysis = analyzer.analyze("Compare the revenue of Company A and B")
        print(analysis.complexity)  # QueryComplexity.MULTI_HOP
        ```
    """
    
    # Keywords indicating multi-hop reasoning
    MULTI_HOP_KEYWORDS: List[str] = [
        "compare", "comparison", "versus", "vs", "difference between",
        "similarities", "contrast", "relationship between", "correlation",
        "how does X affect Y", "impact on", "influence on", "effect of",
        "connection between", "link between", "associated with",
    ]
    
    # Keywords indicating quadratic complexity
    QUADRATIC_KEYWORDS: List[str] = [
        "summarize all", "aggregate", "total", "sum of", "average of",
        "count all", "statistics", "metrics", "analyze everything",
        "comprehensive", "overall", "in total", "complete picture",
        "summary of entire", "key findings across", "main themes in",
    ]
    
    # Keywords indicating simple queries
    SIMPLE_KEYWORDS: List[str] = [
        "what is", "who is", "when did", "where is", "how many",
        "define", "explain", "describe", "list", "name",
    ]
    
    # Keywords indicating context needs
    CONTEXT_KEYWORDS: Set[str] = {
        "entire", "whole", "complete", "full", "all parts",
        "comprehensive", "thorough", "detailed", "extensive",
    }
    
    def __init__(self) -> None:
        """Initialize the query analyzer."""
        logger.info("query_analyzer_initialized")
    
    def analyze(self, query: str) -> QueryAnalysis:
        """Analyze a query and return complexity assessment.
        
        Args:
            query: The user's query string
            
        Returns:
            QueryAnalysis with complexity classification
        """
        query_lower = query.lower()
        
        # Extract keywords found in query
        keywords = self._extract_keywords(query_lower)
        
        # Check for complexity indicators
        is_multi_hop = self._has_multi_hop_keywords(query_lower)
        is_quadratic = self._has_quadratic_keywords(query_lower)
        requires_comparison = self._has_comparison_keywords(query_lower)
        requires_aggregation = self._has_aggregation_keywords(query_lower)
        
        # Determine complexity
        complexity, complexity_score = self._determine_complexity(
            query_lower,
            is_multi_hop,
            is_quadratic,
            requires_comparison,
            requires_aggregation,
        )
        
        # Determine context needs
        context_needs = self._determine_context_needs(query_lower, complexity)
        
        # Estimate reasoning depth
        reasoning_depth = self._estimate_reasoning_depth(
            complexity, is_multi_hop, requires_comparison, requires_aggregation
        )
        
        # Estimate chunk access
        estimated_chunks = self._estimate_chunk_access(
            complexity, context_needs
        )
        
        analysis = QueryAnalysis(
            query=query,
            complexity=complexity,
            complexity_score=complexity_score,
            keywords=keywords,
            context_needs=context_needs,
            reasoning_depth=reasoning_depth,
            is_multi_hop=is_multi_hop,
            requires_comparison=requires_comparison,
            requires_aggregation=requires_aggregation,
            estimated_chunk_access=estimated_chunks,
        )
        
        logger.debug(
            "query_analyzed",
            query=query[:50],
            complexity=complexity.value,
            score=complexity_score,
            multi_hop=is_multi_hop,
        )
        
        return analysis
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract significant keywords from query."""
        # Simple extraction - could be enhanced with NLP
        words = re.findall(r'\b[a-zA-Z]{4,}\b', query)
        
        # Filter out common stop words
        stop_words = {
            "what", "when", "where", "which", "who", "whom", "whose",
            "why", "how", "that", "this", "these", "those", "with",
            "from", "they", "have", "were", "been", "said", "each",
            "will", "about", "could", "would", "should", "there",
        }
        
        keywords = [w for w in words if w.lower() not in stop_words]
        return list(set(keywords))[:10]  # Return top 10 unique keywords
    
    def _has_multi_hop_keywords(self, query: str) -> bool:
        """Check if query contains multi-hop keywords."""
        return any(kw in query for kw in self.MULTI_HOP_KEYWORDS)
    
    def _has_quadratic_keywords(self, query: str) -> bool:
        """Check if query contains quadratic complexity keywords."""
        return any(kw in query for kw in self.QUADRATIC_KEYWORDS)
    
    def _has_comparison_keywords(self, query: str) -> bool:
        """Check if query requires comparison."""
        comparison_patterns = [
            r'\bcompare\w*\b', r'\bversus\b', r'\bvs\b',
            r'\bdifference\w*\b', r'\bsimilarit\w*\b',
            r'\bcontrast\w*\b', r'\bbetter than\b',
            r'\bworse than\b', r'\bsuperior\w*\b',
        ]
        return any(re.search(pattern, query) for pattern in comparison_patterns)
    
    def _has_aggregation_keywords(self, query: str) -> bool:
        """Check if query requires aggregation."""
        aggregation_patterns = [
            r'\bsummar\w*\b', r'\baggregat\w*\b', r'\btotal\w*\b',
            r'\bsum of\b', r'\baverage\w*\b', r'\bcount\w*\b',
            r'\ball\s+\w+', r'\bevery\w*\b', r'\boverall\b',
        ]
        return any(re.search(pattern, query) for pattern in aggregation_patterns)
    
    def _determine_complexity(
        self,
        query: str,
        is_multi_hop: bool,
        is_quadratic: bool,
        requires_comparison: bool,
        requires_aggregation: bool,
    ) -> Tuple[QueryComplexity, float]:
        """Determine query complexity and confidence score.
        
        Returns:
            Tuple of (complexity, confidence_score)
        """
        # Quadratic takes precedence
        if is_quadratic or requires_aggregation:
            # Check if it's also multi-hop
            if is_multi_hop:
                return QueryComplexity.QUADRATIC, 0.85
            return QueryComplexity.QUADRATIC, 0.75
        
        # Multi-hop
        if is_multi_hop or requires_comparison:
            return QueryComplexity.MULTI_HOP, 0.80
        
        # Check for simple patterns
        simple_indicators = sum(1 for kw in self.SIMPLE_KEYWORDS if kw in query)
        if simple_indicators >= 1 and len(query.split()) < 15:
            return QueryComplexity.SIMPLE, 0.70
        
        # Default to single-hop
        return QueryComplexity.SINGLE_HOP, 0.60
    
    def _determine_context_needs(
        self,
        query: str,
        complexity: QueryComplexity,
    ) -> str:
        """Determine context requirements."""
        # Check for extensive context keywords
        if any(kw in query for kw in self.CONTEXT_KEYWORDS):
            return "extensive"
        
        # Based on complexity
        if complexity == QueryComplexity.QUADRATIC:
            return "extensive"
        elif complexity == QueryComplexity.MULTI_HOP:
            return "moderate"
        elif complexity == QueryComplexity.SINGLE_HOP:
            return "moderate"
        else:
            return "minimal"
    
    def _estimate_reasoning_depth(
        self,
        complexity: QueryComplexity,
        is_multi_hop: bool,
        requires_comparison: bool,
        requires_aggregation: bool,
    ) -> int:
        """Estimate reasoning steps needed."""
        base_depth = {
            QueryComplexity.SIMPLE: 1,
            QueryComplexity.SINGLE_HOP: 2,
            QueryComplexity.MULTI_HOP: 3,
            QueryComplexity.QUADRATIC: 4,
        }.get(complexity, 2)
        
        # Add for complexity indicators
        if requires_comparison:
            base_depth += 1
        if requires_aggregation:
            base_depth += 1
        
        return min(base_depth, 6)  # Cap at 6
    
    def _estimate_chunk_access(
        self,
        complexity: QueryComplexity,
        context_needs: str,
    ) -> int:
        """Estimate number of chunks to access."""
        base_chunks = {
            QueryComplexity.SIMPLE: 1,
            QueryComplexity.SINGLE_HOP: 3,
            QueryComplexity.MULTI_HOP: 10,
            QueryComplexity.QUADRATIC: 50,
        }.get(complexity, 5)
        
        # Adjust for context needs
        multipliers = {
            "minimal": 0.5,
            "moderate": 1.0,
            "extensive": 3.0,
        }
        
        return int(base_chunks * multipliers.get(context_needs, 1.0))
    
    def get_complexity_description(self, complexity: QueryComplexity) -> str:
        """Get human-readable description of complexity."""
        descriptions = {
            QueryComplexity.SIMPLE: "Simple fact retrieval",
            QueryComplexity.SINGLE_HOP: "Single-step retrieval and answer",
            QueryComplexity.MULTI_HOP: "Multi-step reasoning required",
            QueryComplexity.QUADRATIC: "Complex aggregation/analysis",
        }
        return descriptions.get(complexity, "Unknown complexity")
