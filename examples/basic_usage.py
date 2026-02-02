"""
Example: Basic RLM Usage

This example demonstrates how to use the RLM Core Engine
to analyze a document and answer questions.
"""

import asyncio
import os

# Set mock environment for testing (replace with real API key for actual usage)
os.environ["RLM_LITELLM_PROVIDER"] = "mock"

from rlm import RLMOrchestrator, MockLLMClient, ChunkedContext


async def basic_example():
    """Basic example with mock LLM."""
    print("=" * 60)
    print("Basic RLM Example")
    print("=" * 60)
    
    # Sample document
    document = """
    Artificial Intelligence (AI) is transforming the world.
    
    Machine learning is a subset of AI that enables computers to learn from data.
    Deep learning is a subset of machine learning using neural networks.
    
    Key applications include:
    - Natural language processing
    - Computer vision
    - Robotics
    - Healthcare diagnostics
    
    The future of AI is promising but requires careful ethical consideration.
    """
    
    # Initialize with mock client (for testing without API keys)
    mock_client = MockLLMClient(
        response_template="""
# Search for relevant information
keywords = ["applications", "future"]
results = []

for keyword in keywords:
    matching = context.search(keyword)
    for idx in matching[:2]:
        chunk = context.get_chunk(idx)
        results.append(f"Found in chunk {idx}: {chunk[:100]}...")

final_answer = "Based on the document, key points are:\n" + "\n".join(results)
print(final_answer)
"""
    )
    
    orchestrator = RLMOrchestrator(llm_client=mock_client)
    
    # Execute query
    result = await orchestrator.execute(
        query="What are the key points about AI in this document?",
        context=document,
    )
    
    print(f"\n✓ Answer: {result.answer}")
    print(f"✓ Execution time: {result.execution_time_ms:.2f}ms")
    print(f"✓ Sub-LLM calls: {result.total_sub_llm_calls}")
    print(f"✓ Session ID: {result.session_id}")
    print(f"✓ Trajectory log: logs/{result.session_id}.jsonl")


async def streaming_example():
    """Example with streaming updates."""
    print("\n" + "=" * 60)
    print("Streaming RLM Example")
    print("=" * 60)
    
    document = "Example document content here..."
    
    mock_client = MockLLMClient(response_template="print('Analysis complete')")
    orchestrator = RLMOrchestrator(llm_client=mock_client)
    
    print("Streaming events:")
    async for event in orchestrator.execute_stream(
        query="Analyze this document",
        context=document,
    ):
        print(f"  [{event.type.value}] {event.data}")


async def large_document_example():
    """Example with large document (simulating 10M tokens)."""
    print("\n" + "=" * 60)
    print("Large Document Example (10M+ tokens)")
    print("=" * 60)
    
    # Create a large document (5000 paragraphs ~ 1M chars ~ 250K tokens)
    paragraphs = []
    for i in range(5000):
        paragraphs.append(f"Section {i}: This is paragraph number {i} with detailed content about topic {i % 10}. It contains important information that may be relevant to queries.")
    
    large_document = "\n\n".join(paragraphs)
    
    print(f"Document size: {len(large_document):,} characters")
    print(f"Estimated tokens: ~{len(large_document) // 4:,}")
    
    # Demonstrate chunked context
    context = ChunkedContext(large_document, chunk_size=100000)
    
    print(f"✓ Document chunked into {context.total_chunks} chunks")
    print(f"✓ Average chunk size: {context.total_length // context.total_chunks:,} chars")
    
    # Search example
    search_results = context.search("section 1000")
    print(f"✓ Search for 'section 1000' found {len(search_results)} matching chunks")


async def main():
    """Run all examples."""
    await basic_example()
    await streaming_example()
    await large_document_example()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
