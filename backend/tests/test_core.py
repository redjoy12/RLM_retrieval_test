"""Basic test for RLM Core Engine."""

import pytest

from rlm import ChunkedContext, MockLLMClient, RLMOrchestrator
from rlm.core.recursion import RecursionController
from rlm.sandbox.local_repl import LocalREPLSandbox


class TestChunkedContext:
    """Test chunked context for large documents."""
    
    def test_chunking_small_document(self):
        """Test chunking with small document."""
        doc = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        context = ChunkedContext(doc, chunk_size=50)
        
        assert context.total_chunks > 0
        assert context.total_length == len(doc)
    
    def test_chunking_large_document(self):
        """Test chunking with large document (simulating 10M tokens)."""
        # Create a large document (10K paragraphs ~ 2M chars ~ 500K tokens)
        paragraphs = [f"This is paragraph {i} with some content." for i in range(10000)]
        doc = "\n\n".join(paragraphs)
        
        context = ChunkedContext(doc, chunk_size=100000)
        
        assert context.total_chunks > 10
        assert context.total_length == len(doc)
        
        # Test chunk retrieval
        chunk_0 = context.get_chunk(0)
        assert len(chunk_0) > 0
        
        # Test search
        results = context.search("paragraph 500")
        assert len(results) > 0
    
    def test_search_functionality(self):
        """Test search in context."""
        doc = "Apple is a fruit.\n\nBanana is yellow.\n\nApple pie is tasty."
        context = ChunkedContext(doc, chunk_size=100)
        
        results = context.search("Apple")
        assert len(results) >= 1


class TestRecursionController:
    """Test recursion controller."""
    
    def test_initialization(self):
        """Test controller initialization."""
        ctrl = RecursionController(max_depth=3, max_total_calls=10)
        assert ctrl.max_depth == 3
        assert ctrl.max_total_calls == 10
    
    def test_root_session(self):
        """Test root session creation."""
        ctrl = RecursionController()
        session_id = ctrl.initialize_root("test query")
        
        assert session_id is not None
        assert len(session_id) > 0
    
    def test_depth_limit(self):
        """Test depth limit enforcement."""
        ctrl = RecursionController(max_depth=2)
        root_id = ctrl.initialize_root("root")
        
        # Should allow first level
        child1 = ctrl.enter_sub_llm(root_id, "child 1")
        assert child1 is not None
        
        # Should allow second level
        child2 = ctrl.enter_sub_llm(child1, "child 2")
        assert child2 is not None
        
        # Should NOT allow third level (depth limit)
        child3 = ctrl.enter_sub_llm(child2, "child 3")
        assert child3 is None
    
    def test_total_call_limit(self):
        """Test total call limit enforcement."""
        ctrl = RecursionController(max_depth=10, max_total_calls=3)
        root_id = ctrl.initialize_root("root")
        
        # Should allow up to 3 calls
        for i in range(3):
            child = ctrl.enter_sub_llm(root_id, f"child {i}")
            assert child is not None
        
        # Fourth call should fail
        child = ctrl.enter_sub_llm(root_id, "child 3")
        assert child is None


class TestLocalSandbox:
    """Test local REPL sandbox."""
    
    @pytest.mark.asyncio
    async def test_simple_execution(self):
        """Test simple code execution."""
        sandbox = LocalREPLSandbox(timeout=5)
        
        def mock_llm(query, chunk):
            return f"Response to: {query}"
        
        result = await sandbox.execute(
            code="print('Hello World')",
            context=None,
            sub_llm_callback=mock_llm,
        )
        
        assert "Hello World" in result.output
        assert result.error is None
    
    @pytest.mark.asyncio
    async def test_sub_llm_call(self):
        """Test sub-LLM call from sandbox."""
        sandbox = LocalREPLSandbox(timeout=5)
        
        call_count = 0
        def mock_llm(query, chunk):
            nonlocal call_count
            call_count += 1
            return f"Response: {query}"
        
        code = """
result = llm_query("Test query", None)
print(result)
"""
        
        result = await sandbox.execute(
            code=code,
            context=None,
            sub_llm_callback=mock_llm,
        )
        
        assert call_count == 1
        assert "Response: Test query" in result.output
    
    @pytest.mark.asyncio
    async def test_timeout(self):
        """Test timeout enforcement."""
        sandbox = LocalREPLSandbox(timeout=1)
        
        def mock_llm(query, chunk):
            return "response"
        
        code = """
import time
time.sleep(10)  # This should timeout
"""
        
        result = await sandbox.execute(
            code=code,
            context=None,
            sub_llm_callback=mock_llm,
        )
        
        assert result.error is not None
        assert "timeout" in result.error.lower()


class TestMockLLMClient:
    """Test mock LLM client."""
    
    @pytest.mark.asyncio
    async def test_generate(self):
        """Test mock generation."""
        client = MockLLMClient(response_template="Echo: {prompt}")
        
        response = await client.generate("Hello")
        
        assert "Echo: Hello" in response.content
        assert response.model == "mock-model"
    
    @pytest.mark.asyncio
    async def test_stream(self):
        """Test mock streaming."""
        client = MockLLMClient(response_template="Hello World")
        
        chunks = []
        async for chunk in client.generate_stream("Test"):
            chunks.append(chunk)
        
        assert len(chunks) > 0
        assert "Hello" in "".join(chunks)


class TestRLMOrchestrator:
    """Test main orchestrator."""
    
    @pytest.mark.asyncio
    async def test_basic_execution(self):
        """Test basic RLM execution with mock."""
        mock_client = MockLLMClient(
            response_template="""
result = llm_query("Analyze", "Test context")
print(result)
"""
        )
        
        orchestrator = RLMOrchestrator(llm_client=mock_client)
        
        result = await orchestrator.execute(
            query="Test query",
            context="Test document content",
        )
        
        assert result.answer is not None
        assert result.session_id is not None
        assert result.execution_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_large_document(self):
        """Test with large document (simulating 10M tokens)."""
        mock_client = MockLLMClient(response_template="print('Analysis complete')")
        
        orchestrator = RLMOrchestrator(llm_client=mock_client)
        
        # Create large document
        paragraphs = [f"Paragraph {i} content here." for i in range(1000)]
        large_doc = "\n\n".join(paragraphs)
        
        result = await orchestrator.execute(
            query="Analyze this large document",
            context=large_doc,
        )
        
        assert result.answer is not None
        # Verify it was chunked
        assert result.metadata["context_chunks"] > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
