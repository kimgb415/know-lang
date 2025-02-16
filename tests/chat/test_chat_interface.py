import pytest
from unittest.mock import Mock, patch
from gradio import ChatMessage
from typing import List

from knowlang.chat_bot.chat_interface import (
    CodeQAChatInterface,
    CodeContext,
)
from knowlang.chat_bot.chat_graph import (
    ChatStatus,
    StreamingChatResult,
    RetrievedContext
)

@pytest.fixture
def mock_request():
    """Mock gradio request object"""
    request = Mock()
    request.request.client.host = "127.0.0.1"
    return request

@pytest.fixture
@patch('knowlang.chat_bot.chat_interface.VectorStoreFactory')
def interface(mock_vector_store_factory, mock_config, mock_vector_store):
    """Create test interface instance with mocked dependencies"""
    mock_vector_store_factory.get.return_value = mock_vector_store
    interface = CodeQAChatInterface(mock_config)
    
    return interface

@pytest.mark.asyncio
@patch('knowlang.chat_bot.chat_interface.stream_chat_progress')
async def test_stream_response_successful_flow(mock_stream_progress, interface, mock_request):
    """Test successful streaming response flow"""
    # Create an async generator using AsyncMock
    async def mock_stream():
        yield StreamingChatResult(
            answer="",
            status=ChatStatus.STARTING,
            progress_message="Processing question"
        )
        yield StreamingChatResult(
            answer="",
            status=ChatStatus.RETRIEVING,
            progress_message="Searching codebase"
        )
        yield StreamingChatResult(
            answer="Final answer",
            status=ChatStatus.COMPLETE,
            progress_message="Complete",
            retrieved_context=RetrievedContext(
                chunks=["def test():\n    pass"],
                metadatas=[{
                    "file_path": "test.py",
                    "start_line": 1,
                    "end_line": 2
                }]
            )
        )
    
    mock_stream_progress.return_value = mock_stream()
    
    history: List[ChatMessage] = []
    message = "test question"
    
    # Collect all streamed responses
    responses = []
    async for updated_history in interface.stream_response(message, history, mock_request):
        responses.append(updated_history)
    
    # Verify response sequence
    assert len(responses) >= 4  # User msg + progress + code context + final answer
    final_history = responses[-1]
    
    # Verify message sequence
    assert final_history[0].content == message  # User question
    assert "```python" in final_history[1].content  # Code context
    assert final_history[2].content == "Final answer"  # Final answer

def test_format_code_block(interface):
    """Test code block formatting"""
    code = "def test():\n    pass"
    metadata = {
        "file_path": "test.py",
        "start_line": 1,
        "end_line": 2
    }
    
    formatted = interface._format_code_block(code, metadata)
    
    # Verify formatting
    assert "📄 test.py (lines 1-2)" in formatted
    assert "```python" in formatted
    assert code in formatted
    assert "</details>" in formatted

def test_code_context_formatting():
    """Test CodeContext class formatting"""
    context = CodeContext(
        file_path="test.py",
        start_line=1,
        end_line=10
    )
    
    assert context.to_title() == "📄 test.py (lines 1-10)"
    
    # Test creation from metadata
    metadata = {
        "file_path": "test.py",
        "start_line": 1,
        "end_line": 10
    }
    from_metadata = CodeContext.from_metadata(metadata)
    assert from_metadata.file_path == context.file_path
    assert from_metadata.start_line == context.start_line
    assert from_metadata.end_line == context.end_line

@pytest.mark.asyncio
async def test_handle_feedback(interface, mock_request):
    """Test feedback handling"""
    # Setup mock history
    history = [
        ChatMessage(role="user", content="test question"),
        ChatMessage(role="assistant", content="test answer")
    ]
    
    # Setup mock like data
    like_data = Mock()
    like_data.index = 1  # Index of response message
    like_data.liked = True
    
    # Mock analytics tracking
    interface.chat_analytics.track_feedback = Mock()
    
    # Handle feedback
    interface._handle_feedback(like_data, history, mock_request)
    
    # Verify analytics called correctly
    interface.chat_analytics.track_feedback.assert_called_once_with(
        like=True,
        query="test question",
        client_ip="127.0.0.1"
    )

@pytest.mark.asyncio
@patch('knowlang.chat_bot.chat_interface.stream_chat_progress')
async def test_stream_response_error_handling(mock_stream_progress, interface, mock_request):
    """Test error handling in stream_response"""
    # Mock stream_chat_progress to raise an error
    async def mock_stream():
        yield Exception("Test error")
    
    # mock_stream_progress.side_effect = mock_stream()
    mock_stream_progress.side_effect = Exception("Test error")

    
    history: List[ChatMessage] = []
    message = "test question"
    
    # Collect all streamed responses
    responses = []
    async for updated_history in interface.stream_response(message, history, mock_request):
        responses.append(updated_history)
    
    # Verify error handling
    final_history = responses[-1]
    assert len(final_history) >= 2  # User message + error message
    assert "Test error" in final_history[-1].content
    assert final_history[-1].metadata.get("status") == "error"

def test_create_interface(interface):
    """Test interface creation"""
    result = interface.create_interface()
    
    # Verify interface components exist
    assert result is not None
    # Note: Since gr.Blocks() creates a complex interface object,
    # we mainly verify it returns without error rather than checking internals