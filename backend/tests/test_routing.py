"""Query Router & Optimizer - Component 7 Test Suite."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from rlm.routing.analyzer import QueryAnalyzer, QueryComplexity
from rlm.routing.cost_estimator import CostEstimator
from rlm.routing.models import (
    ExecutionStrategy,
    RoutingDecision,
    QueryAnalysis,
    CostEstimate,
)
from rlm.routing.optimizer import QueryOptimizer, SimpleOptimizer
from rlm.routing.query_router import QueryRouter
from rlm.routing.strategies import StrategyFactory


class TestQueryAnalyzer:
    """Test suite for QueryAnalyzer."""
    
    def test_simple_query_detection(self):
        """Test detection of simple queries."""
        analyzer = QueryAnalyzer()
        
        queries = [
            "What is machine learning?",
            "Who wrote this document?",
            "When was this published?",
        ]
        
        for query in queries:
            analysis = analyzer.analyze(query)
            assert analysis.complexity in [QueryComplexity.SIMPLE, QueryComplexity.SINGLE_HOP]
            assert analysis.reasoning_depth <= 2
    
    def test_multi_hop_detection(self):
        """Test detection of multi-hop queries."""
        analyzer = QueryAnalyzer()
        
        queries = [
            "Compare the revenue of Company A and B",
            "What is the relationship between X and Y?",
            "How does inflation affect stock prices?",
        ]
        
        for query in queries:
            analysis = analyzer.analyze(query)
            assert analysis.is_multi_hop or analysis.requires_comparison
            assert analysis.complexity in [QueryComplexity.MULTI_HOP, QueryComplexity.QUADRATIC]
    
    def test_quadratic_detection(self):
        """Test detection of quadratic/complex queries."""
        analyzer = QueryAnalyzer()
        
        queries = [
            "Summarize all the key findings",
            "Analyze everything in the document",
            "What are the main themes across all sections?",
        ]
        
        for query in queries:
            analysis = analyzer.analyze(query)
            assert analysis.complexity == QueryComplexity.QUADRATIC or analysis.requires_aggregation
    
    def test_keyword_extraction(self):
        """Test keyword extraction."""
        analyzer = QueryAnalyzer()
        
        query = "What are the differences between machine learning and deep learning?"
        analysis = analyzer.analyze(query)
        
        assert len(analysis.keywords) > 0
        assert any("learning" in kw.lower() for kw in analysis.keywords)
    
    def test_complexity_descriptions(self):
        """Test complexity descriptions."""
        analyzer = QueryAnalyzer()
        
        descriptions = {
            QueryComplexity.SIMPLE: "Simple",
            QueryComplexity.SINGLE_HOP: "Single",
            QueryComplexity.MULTI_HOP: "Multi",
            QueryComplexity.QUADRATIC: "Quadratic",
        }
        
        for complexity, expected in descriptions.items():
            desc = analyzer.get_complexity_description(complexity)
            assert expected in desc


class TestCostEstimator:
    """Test suite for CostEstimator."""
    
    def test_cost_estimation_structure(self):
        """Test cost estimate structure."""
        estimator = CostEstimator()
        
        estimate = estimator.estimate_cost(
            query="What is AI?",
            context="AI is artificial intelligence...",
            model="gpt-5-mini",
        )
        
        assert isinstance(estimate, CostEstimate)
        assert estimate.estimated_input_tokens > 0
        assert estimate.estimated_output_tokens > 0
        assert estimate.estimated_total_tokens > 0
        assert estimate.estimated_cost_usd >= 0
        assert estimate.cost_with_buffer >= estimate.estimated_cost_usd
    
    def test_rag_cost_estimation(self):
        """Test RAG-specific cost estimation."""
        estimator = CostEstimator()
        
        estimate = estimator.estimate_rag_cost(
            query="What are the findings?",
            num_chunks=5,
            model="gpt-5-mini",
        )
        
        assert estimate.estimated_input_tokens > 0
        assert estimate.estimated_output_tokens > 0
    
    def test_rlm_cost_estimation(self):
        """Test RLM-specific cost estimation."""
        estimator = CostEstimator()
        
        estimate = estimator.estimate_rlm_cost(
            query="Analyze the trends",
            context_size=100000,
            estimated_sub_llm_calls=10,
            model="gpt-5-mini",
        )
        
        # RLM should have higher cost due to multiple calls
        assert estimate.estimated_input_tokens > 0
        assert estimate.estimated_output_tokens > 0
    
    def test_strategy_comparison(self):
        """Test cost comparison across strategies."""
        estimator = CostEstimator()
        
        comparisons = estimator.compare_strategies(
            query="What is AI?",
            context_size=50000,
            model="gpt-5-mini",
        )
        
        assert "direct_llm" in comparisons
        assert "rag" in comparisons
        assert "rlm" in comparisons
        assert "hybrid" in comparisons
        
        # All should have cost estimates
        for strategy, estimate in comparisons.items():
            assert estimate.estimated_cost_usd >= 0
    
    def test_model_pricing_lookup(self):
        """Test model pricing lookup."""
        estimator = CostEstimator()
        
        # Test known models
        pricing = estimator._get_model_pricing("gpt-5-mini")
        assert "input" in pricing
        assert "output" in pricing
        
        # Test unknown model returns default
        pricing = estimator._get_model_pricing("unknown-model")
        assert "input" in pricing
        assert "output" in pricing


class TestQueryOptimizer:
    """Test suite for QueryOptimizer."""
    
    @pytest.mark.asyncio
    async def test_simple_optimizer(self):
        """Test simple rule-based optimizer."""
        optimizer = SimpleOptimizer()
        
        result = optimizer.optimize("Tell me everything about this")
        
        assert result.original_query == "Tell me everything about this"
        assert len(result.suggestions) > 0
        assert not result.use_optimized
    
    @pytest.mark.asyncio
    async def test_optimizer_with_complexity(self):
        """Test optimizer with complexity information."""
        optimizer = QueryOptimizer()
        
        # Mock LLM client
        optimizer.llm_client = MagicMock()
        optimizer.llm_client.generate = AsyncMock(return_value=MagicMock(
            content='{"reformulation": "What is machine learning?", "explanation": "Clearer phrasing"}'
        ))
        
        result = await optimizer.optimize(
            "What ML stuff is there?",
            QueryComplexity.SIMPLE,
        )
        
        assert result.original_query == "What ML stuff is there?"


class TestQueryRouter:
    """Test suite for QueryRouter."""
    
    @pytest.mark.asyncio
    async def test_strategy_selection_logic(self):
        """Test strategy selection based on complexity and context."""
        router = QueryRouter()
        
        # Mock context sizes
        with patch.object(router, '_get_context_size', return_value=5000):
            with patch.object(router.analyzer, 'analyze', return_value=QueryAnalysis(
                query="What is AI?",
                complexity=QueryComplexity.SIMPLE,
                complexity_score=0.8,
                keywords=["AI"],
                context_needs="minimal",
                reasoning_depth=1,
                is_multi_hop=False,
                requires_comparison=False,
                requires_aggregation=False,
                estimated_chunk_access=1,
            )):
                decision = await router.analyze_query("What is AI?", ["doc-123"])
                
                assert decision.strategy == ExecutionStrategy.DIRECT_LLM
                assert decision.confidence > 0.5
    
    def test_routing_decision_structure(self):
        """Test routing decision has all required fields."""
        decision = RoutingDecision(
            query="Test query",
            strategy=ExecutionStrategy.RAG,
            confidence=0.85,
            reasoning="Test reasoning",
            query_analysis=QueryAnalysis(
                query="Test",
                complexity=QueryComplexity.SINGLE_HOP,
                complexity_score=0.8,
                keywords=["test"],
                context_needs="moderate",
                reasoning_depth=2,
                is_multi_hop=False,
                requires_comparison=False,
                requires_aggregation=False,
                estimated_chunk_access=3,
            ),
            cost_estimate=CostEstimate(
                estimated_input_tokens=1000,
                estimated_output_tokens=500,
                estimated_total_tokens=1500,
                estimated_cost_usd=0.001,
                model_used="gpt-5-mini",
            ),
            document_ids=["doc-123"],
            context_size=50000,
            estimated_chunks=5,
        )
        
        # Test serialization
        data = decision.to_dict()
        assert "query" in data
        assert "strategy" in data
        assert "cost_estimate" in data
    
    def test_create_visibility(self):
        """Test routing visibility creation."""
        router = QueryRouter()
        
        decision = RoutingDecision(
            query="Test",
            strategy=ExecutionStrategy.HYBRID,
            confidence=0.88,
            reasoning="Large context with complex query",
            query_analysis=QueryAnalysis(
                query="Test",
                complexity=QueryComplexity.MULTI_HOP,
                complexity_score=0.8,
                keywords=["test"],
                context_needs="extensive",
                reasoning_depth=3,
                is_multi_hop=True,
                requires_comparison=True,
                requires_aggregation=False,
                estimated_chunk_access=10,
            ),
            cost_estimate=CostEstimate(
                estimated_input_tokens=5000,
                estimated_output_tokens=1000,
                estimated_total_tokens=6000,
                estimated_cost_usd=0.01,
                model_used="gpt-5-mini",
            ),
            document_ids=["doc-123"],
            context_size=600000,
            estimated_chunks=10,
        )
        
        visibility = router._create_visibility(decision)
        
        assert visibility.strategy_name == "HYBRID"
        assert visibility.confidence_percent == 88
        assert "$0.01" in visibility.estimated_cost_usd


class TestStrategyFactory:
    """Test suite for StrategyFactory."""
    
    def test_create_direct_llm_strategy(self):
        """Test creating direct LLM strategy."""
        from rlm.routing.strategies import DirectLLMStrategy
        
        strategy = StrategyFactory.create_strategy(ExecutionStrategy.DIRECT_LLM)
        
        assert isinstance(strategy, DirectLLMStrategy)
        assert strategy.name == "Direct LLM"
    
    def test_create_rag_strategy(self):
        """Test creating RAG strategy."""
        from rlm.routing.strategies import RAGStrategy
        
        strategy = StrategyFactory.create_strategy(ExecutionStrategy.RAG)
        
        assert isinstance(strategy, RAGStrategy)
        assert strategy.name == "RAG"
    
    def test_create_rlm_strategy(self):
        """Test creating RLM strategy."""
        from rlm.routing.strategies import RLMStrategy
        
        strategy = StrategyFactory.create_strategy(ExecutionStrategy.RLM)
        
        assert isinstance(strategy, RLMStrategy)
        assert strategy.name == "RLM"
    
    def test_create_hybrid_strategy(self):
        """Test creating hybrid strategy."""
        from rlm.routing.strategies import HybridStrategy
        
        strategy = StrategyFactory.create_strategy(ExecutionStrategy.HYBRID)
        
        assert isinstance(strategy, HybridStrategy)
        assert strategy.name == "Hybrid RAG+RLM"
    
    def test_unknown_strategy_raises_error(self):
        """Test that unknown strategy raises error."""
        with pytest.raises(ValueError):
            StrategyFactory.create_strategy("unknown_strategy")


class TestRoutingModels:
    """Test suite for routing data models."""
    
    def test_cost_estimate_with_buffer(self):
        """Test cost estimate buffer calculation."""
        estimate = CostEstimate(
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
            estimated_total_tokens=1500,
            estimated_cost_usd=0.001,
            model_used="gpt-5-mini",
            cost_buffer_percent=10,
        )
        
        assert estimate.cost_with_buffer == 0.0011  # 0.001 * 1.1
    
    def test_query_complexity_enum(self):
        """Test query complexity enum values."""
        assert QueryComplexity.SIMPLE.value == "simple"
        assert QueryComplexity.SINGLE_HOP.value == "single_hop"
        assert QueryComplexity.MULTI_HOP.value == "multi_hop"
        assert QueryComplexity.QUADRATIC.value == "quadratic"
    
    def test_execution_strategy_enum(self):
        """Test execution strategy enum values."""
        assert ExecutionStrategy.DIRECT_LLM.value == "direct_llm"
        assert ExecutionStrategy.RAG.value == "rag"
        assert ExecutionStrategy.RLM.value == "rlm"
        assert ExecutionStrategy.HYBRID.value == "hybrid"


# Integration tests (optional, for full system testing)
@pytest.mark.integration
class TestRoutingIntegration:
    """Integration tests for full routing pipeline."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_simple_query(self):
        """Test end-to-end routing for simple query."""
        router = QueryRouter()
        
        # This would require mocked document storage
        # For now, just test the analysis phase
        with patch.object(router, '_get_context_size', return_value=5000):
            decision = await router.analyze_query(
                "What is this document about?",
                ["doc-123"]
            )
            
            assert decision is not None
            assert decision.query == "What is this document about?"
            assert decision.cost_estimate is not None
    
    @pytest.mark.asyncio
    async def test_cost_estimation_accuracy(self):
        """Test that cost estimates are reasonable."""
        estimator = CostEstimator()
        
        # Estimate for a typical query
        estimate = estimator.estimate_cost(
            query="Summarize the main points",
            context="This is a document with some content..." * 100,
            model="gpt-5-mini",
        )
        
        # Should be non-zero but reasonable
        assert estimate.estimated_cost_usd > 0
        assert estimate.estimated_cost_usd < 1.0  # Should be less than $1 for typical query
        assert estimate.estimated_total_tokens > 0
        assert estimate.estimated_total_tokens < 100000  # Should be reasonable


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
