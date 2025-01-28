import pytest
import tempfile
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
from pydantic_ai import Agent
from know_lang_bot.code_parser.summarizer import CodeSummarizer
from know_lang_bot.code_parser.parser import CodeChunk, ChunkType
from know_lang_bot.config import AppConfig

@pytest.fixture
def config():
    """Create a test configuration"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield AppConfig(
            llm={"model_name": "test-model", "model_provider": "test"},
            db={"persist_directory": Path(temp_dir), "collection_name": "test_collection"}
        )

@pytest.fixture
def sample_chunks():
    """Create sample code chunks for testing"""
    return [
        CodeChunk(
            type=ChunkType.FUNCTION,
            content="def hello(): return 'world'",
            start_line=1,
            end_line=2,
            file_path="test.py",
            name="hello",
            docstring="Says hello"
        ),
        CodeChunk(
            type=ChunkType.CLASS,
            content="class TestClass:\n    def __init__(self):\n        pass",
            start_line=4,
            end_line=6,
            file_path="test.py",
            name="TestClass",
            docstring="A test class"
        )
    ]

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

@pytest.mark.asyncio
@patch('know_lang_bot.code_parser.summarizer.Agent')
async def test_summarize_chunk(mock_agent_class, config: AppConfig, sample_chunks: list[CodeChunk], mock_run_result: Mock):
    """Test summarizing a single chunk"""
    # Setup the mock agent instance
    mock_agent = mock_agent_class.return_value
    mock_agent.run = AsyncMock(return_value=mock_run_result)

    summarizer = CodeSummarizer(config)
    result = await summarizer.summarize_chunk(sample_chunks[0])
    
    # Verify result
    assert isinstance(result, str)
    assert result == mock_run_result.data
    
    # Verify agent was called with correct prompt
    call_args = mock_agent.run.call_args[0][0]
    assert "def hello()" in call_args
    assert "Says hello" in call_args

@patch('know_lang_bot.code_parser.summarizer.Agent')
def test_chromadb_initialization(mock_agent_class, config: AppConfig):
    """Test ChromaDB initialization"""
    mock_agent = mock_agent_class.return_value
    
    summarizer = CodeSummarizer(config)
    assert summarizer.collection is not None
    
    # Verify we can create a new collection
    summarizer.db_client.delete_collection(config.db.collection_name)
    new_summarizer = CodeSummarizer(config)
    assert new_summarizer.collection is not None