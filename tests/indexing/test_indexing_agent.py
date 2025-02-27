import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from knowlang.configs import AppConfig
from knowlang.core.types import (BaseChunkType, CodeChunk, CodeLocation,
                                 LanguageEnum)
from knowlang.indexing.indexing_agent import IndexingAgent
from knowlang.utils import format_code_summary
from knowlang.vector_stores import VectorStoreError
from knowlang.vector_stores.mock import MockVectorStore


@pytest.fixture
def config():
    """Create a test configuration"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield AppConfig(
            llm={"model_name": "testing", "model_provider": "testing"},
            db={"persist_directory": Path(temp_dir), "collection_name": "test_collection"}
        )

@pytest.fixture
def sample_chunks():
    """Create sample code chunks for testing"""
    return [
        CodeChunk(
            language=LanguageEnum.PYTHON,
            type=BaseChunkType.FUNCTION,
            content="def hello(): return 'world'",
            location=CodeLocation(
                start_line=1,
                end_line=2,
                file_path="test.py"
            ),
            name="hello",
            docstring="Says hello"
        ),
        CodeChunk(
            language=LanguageEnum.PYTHON,
            type=BaseChunkType.CLASS,
            content="class TestClass:\n    def __init__(self):\n        pass",
            location=CodeLocation(
                start_line=4,
                end_line=6,
                file_path="test.py"
            ),
            name="TestClass",
            docstring="A test class"
        )
    ]

@pytest.fixture
def mock_vector_store():
    """Create a mock vector store"""
    store = MockVectorStore()
    return store

@pytest.fixture
def mock_summary():
    """Create a sample summary result"""
    return "This is a test function"

@pytest.fixture
def mock_run_result(mock_summary):
    """Create a mock run result"""
    mock_result = Mock()
    mock_result.data = mock_summary
    return mock_result

@pytest.fixture
@patch('knowlang.indexing.indexing_agent.VectorStoreFactory')
def indexing_agent(mock_vector_store_factory, config: AppConfig, mock_vector_store: MockVectorStore):
    """Create a mock indexing_agent instance"""
    mock_vector_store_factory.get.return_value = mock_vector_store
    return IndexingAgent(config)

@pytest.mark.asyncio
@patch('knowlang.indexing.indexing_agent.Agent')
async def test_summarize_chunk(
    mock_agent_class, 
    config: AppConfig, 
    sample_chunks: list[CodeChunk], 
    mock_run_result: Mock,
    mock_vector_store: MockVectorStore,
    indexing_agent: IndexingAgent
):
    """Test summarizing a single chunk"""
    # Setup the mock agent instance
    mock_agent = mock_agent_class.return_value
    mock_agent.run = AsyncMock(return_value=mock_run_result)

    indexing_agent.agent = mock_agent
    result = await indexing_agent.summarize_chunk(sample_chunks[0])
    
    # Verify result
    assert isinstance(result, str)
    assert result == format_code_summary(sample_chunks[0].content, mock_run_result.data)
    
    # Verify agent was called with correct prompt
    call_args = mock_agent.run.call_args[0][0]
    assert "def hello()" in call_args
    assert "Says hello" in call_args
