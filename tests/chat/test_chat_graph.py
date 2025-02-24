# test_chat_graph.py
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic_graph import End, GraphRunContext
from voyageai.object.reranking import RerankingResult

from knowlang.chat_bot.chat_graph import (AnswerQuestionNode, ChatGraphDeps,
                                          ChatGraphState, ChatResult,
                                          ChatStatus, PolishQuestionNode,
                                          RetrieveContextNode,
                                          RetrievedContext, SearchResult,
                                          stream_chat_progress)


@pytest.mark.asyncio
@patch('knowlang.chat_bot.chat_graph.Agent')
async def test_polish_question_node(mock_agent_class, mock_config, mock_vector_store):
    """Test that PolishQuestionNode properly refines questions"""
    node = PolishQuestionNode()
    state = ChatGraphState(original_question="how does this work?")
    deps = ChatGraphDeps(vector_store=mock_vector_store, config=mock_config)
    ctx = GraphRunContext(state=state, deps=deps)
    
    mock_answer = Mock()
    mock_answer.data = "How does the implementation of this function work in the codebase?"
    mock_agent = mock_agent_class.return_value
    mock_agent.run = AsyncMock(return_value=mock_answer)

    next_node = await node.run(ctx)
    assert isinstance(next_node, RetrieveContextNode)
    assert ctx.state.polished_question == "How does the implementation of this function work in the codebase?"
    mock_agent.run.assert_called_once()

@pytest.mark.asyncio
@patch('knowlang.chat_bot.chat_graph.generate_embedding')
@patch('knowlang.chat_bot.chat_graph.voyageai')
async def test_retrieve_context_node_success(
    mock_voyageai, 
    mock_embedding_generator, 
    mock_config, 
    populated_mock_store
):
    """Test successful context retrieval with reranking"""
    node = RetrieveContextNode()
    state = ChatGraphState(
        original_question="test question",
        polished_question="refined test question"
    )
    deps = ChatGraphDeps(vector_store=populated_mock_store, config=mock_config)
    ctx = GraphRunContext(state=state, deps=deps)

    # Mock embedding generation
    mock_embedding = [1.0, 0.0, 0.0]  # This will match first test document
    mock_embedding_generator.return_value = mock_embedding
    mock_voyage_client = mock_voyageai.Client.return_value
    mock_rerank_obj = Mock()
    mock_rerank_obj.results = [
        RerankingResult(relevance_score=0.8, document="def test_function(): pass", index=0),
        RerankingResult(relevance_score=0.6, document="class TestClass: pass", index=1)
    ]

    mock_rerank = AsyncMock(return_value=mock_rerank_obj)
    mock_voyage_client.rerank = mock_rerank

    populated_mock_store.search = AsyncMock(return_value=[
        # Create SearchResult objects with appropriate fields.
        SearchResult(document="def test_function(): pass", metadata={"file_path": "test1.py", "start_line": 1, "end_line": 2}, score=1.0),
        SearchResult(document="class TestClass: pass", metadata={"file_path": "test2.py", "start_line": 10, "end_line": 12}, score=0.5)
    ])

    next_node = await node.run(ctx)

    assert isinstance(next_node, AnswerQuestionNode)
    assert ctx.state.retrieved_context is not None
    assert len(ctx.state.retrieved_context.chunks) == 1  # Only one chunk above threshold
    assert ctx.state.retrieved_context.chunks[0] == "def test_function(): pass"

@pytest.mark.asyncio
@patch('knowlang.chat_bot.chat_graph.Agent')
async def test_answer_question_node(mock_agent_class, mock_config, populated_mock_store):
    """Test that AnswerQuestionNode generates appropriate answers"""
    node = AnswerQuestionNode()
    state = ChatGraphState(
        original_question="test question",
        retrieved_context=RetrievedContext(
            chunks=["def test_function(): pass"],
            metadatas=[{"file_path": "test1.py", "start_line": 1, "end_line": 2}]
        )
    )
    deps = ChatGraphDeps(vector_store=populated_mock_store, config=mock_config)
    ctx = GraphRunContext(state=state, deps=deps)

    mock_answer = Mock()
    mock_answer.data = "This is the answer based on the code context."
    mock_agent = mock_agent_class.return_value
    mock_agent.run = AsyncMock(return_value=mock_answer)

    result = await node.run(ctx)
    assert result.data.answer == "This is the answer based on the code context."
    assert result.data.retrieved_context == state.retrieved_context
    mock_agent.run.assert_called_once()


@pytest.mark.asyncio
async def test_answer_question_node_no_context(mock_config, mock_vector_store):
    """Test AnswerQuestionNode behavior when no context is found"""
    node = AnswerQuestionNode()
    state = ChatGraphState(
        original_question="test question",
        retrieved_context=RetrievedContext(chunks=[], metadatas=[])
    )
    deps = ChatGraphDeps(vector_store=mock_vector_store, config=mock_config)
    ctx = GraphRunContext(state=state, deps=deps)

    
    result = await node.run(ctx)
    assert "couldn't find any relevant code context" in result.data.answer.lower()
    assert result.data.retrieved_context is None


@pytest.mark.asyncio
@patch('knowlang.chat_bot.chat_graph.logfire')
@patch('knowlang.chat_bot.chat_graph.chat_graph')
async def test_stream_chat_progress_success(
    mock_chat_graph, 
    mock_logfire, 
    mock_config, 
    populated_mock_store
):
    """Test successful streaming chat progress with all stages"""
    # Mock the span context manager
    mock_span = Mock()
    mock_logfire.span.return_value.__enter__.return_value = mock_span
    
    # Set up mock chat graph behavior
    mock_chat_graph.next = AsyncMock()
    mock_chat_graph.next.side_effect = [
        # StreamingChatResponses are yield before each Node's next() is called
        AnswerQuestionNode(),   # Move to answer node
        End(ChatResult(         # Finally return the result
            answer="Test answer",
            retrieved_context=RetrievedContext(
                chunks=["def test_function(): pass"],
                metadatas=[{"file_path": "test1.py", "start_line": 1, "end_line": 2}]
            )
        ))
    ]

    # Collect all streamed results
    results = []
    async for result in stream_chat_progress(
        question="test question",
        vector_store=populated_mock_store,
        config=mock_config
    ):
        results.append(result)

    # Verify the sequence of streaming results
    assert len(results) >= 4  # Should have all stages
    
    # Verify initial status
    assert results[0].status == ChatStatus.STARTING
    assert "Processing question: test question" in results[0].progress_message
    
    # Verify retrieval status
    assert results[1].status == ChatStatus.RETRIEVING
    assert "Searching codebase" in results[1].progress_message
    
    # Verify answering status
    assert results[2].status == ChatStatus.ANSWERING
    assert "Generating answer" in results[2].progress_message
    
    # Verify final result
    assert results[-1].status == ChatStatus.COMPLETE
    assert results[-1].answer == "Test answer"
    assert results[-1].retrieved_context is not None
    
    # Verify graph execution
    assert mock_chat_graph.next.call_count == 2
    mock_span.set_attribute.assert_called_once()

@pytest.mark.asyncio
@patch('knowlang.chat_bot.chat_graph.logfire')
@patch('knowlang.chat_bot.chat_graph.chat_graph')
async def test_stream_chat_progress_node_error(mock_chat_graph, mock_logfire, mock_vector_store, mock_config):
    """Test streaming chat progress when a node execution fails"""
    # Mock the span context manager
    mock_span = Mock()
    mock_logfire.span.return_value.__enter__.return_value = mock_span
    
    # Set up mock chat graph to raise an error
    mock_chat_graph.next = AsyncMock()
    mock_chat_graph.next.side_effect = Exception("Test node error")

    # Collect all streamed results
    results = []
    async for result in stream_chat_progress(
        question="test question",
        vector_store=mock_vector_store,
        config=mock_config
    ):
        results.append(result)

    # Verify error handling
    assert len(results) == 3  # Should have initial status, pending first node, and error
    
    # Verify initial status
    assert results[0].status == ChatStatus.STARTING
    
    # Verify error status
    assert results[-1].status == ChatStatus.ERROR
    assert "Test node error" in results[-1].progress_message
    assert not results[-1].retrieved_context

@pytest.mark.asyncio
@patch('knowlang.chat_bot.chat_graph.logfire')
@patch('knowlang.chat_bot.chat_graph.chat_graph')
async def test_stream_chat_progress_invalid_node(mock_chat_graph, mock_logfire, mock_vector_store, mock_config):
    """Test streaming chat progress when an invalid node type is returned"""
    # Mock the span context manager
    mock_span = Mock()
    mock_logfire.span.return_value.__enter__.return_value = mock_span
    
    # Set up mock chat graph to return invalid node type
    mock_chat_graph.next = AsyncMock()
    mock_chat_graph.next.return_value = "invalid node"  # Return invalid node type

    # Collect all streamed results
    results = []
    async for result in stream_chat_progress(
        question="test question",
        vector_store=mock_vector_store,
        config=mock_config
    ):
        results.append(result)

    # Verify error handling
    assert len(results) == 3  # Should have initial status , pending first node, and error
    assert results[-1].status == ChatStatus.ERROR
    assert "Invalid node type" in results[-1].progress_message

@pytest.mark.asyncio
@patch('knowlang.chat_bot.chat_graph.logfire')
async def test_stream_chat_progress_general_error(mock_logfire, mock_vector_store, mock_config):
    """Test streaming chat progress when a general error occurs"""
    # Mock the span context manager to raise an error
    mock_logfire.span.side_effect = Exception("Test general error")

    # Collect all streamed results
    results = []
    async for result in stream_chat_progress(
        question="test question",
        vector_store=mock_vector_store,
        config=mock_config
    ):
        results.append(result)

    # Verify error handling
    assert len(results) == 2  # Should have initial status and error
    assert results[-1].status == ChatStatus.ERROR
    assert "Test general error" in results[-1].progress_message