import pytest
import tempfile
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
from knowlang.summarizer.summarizer import CodeSummarizer
from knowlang.core.types import CodeChunk, ChunkType
from knowlang.configs.config import AppConfig
from knowlang.utils.chunking_util import format_code_summary

@pytest.fixture
def config():
    """Create a test configuration"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield AppConfig(
            llm={"model_name": "testing", "model_provider": "testing"},
            db={"persist_directory": Path(temp_dir), "collection_name": "test_collection"}
        )

@pytest.fixture
def sample_chunks(config: AppConfig):
    """Create sample code chunks for testing"""
    return [
        CodeChunk(
            type=ChunkType.FUNCTION,
            content="def hello(): return 'world'",
            start_line=1,
            end_line=2,
            file_path=str(config.db.codebase_directory / "test.py"),
            name="hello",
            docstring="Says hello"
        ),
        CodeChunk(
            type=ChunkType.CLASS,
            content="class TestClass:\n    def __init__(self):\n        pass",
            start_line=4,
            end_line=6,
            file_path=str(config.db.codebase_directory / "test.py"),
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
@patch('knowlang.summarizer.summarizer.Agent')
async def test_summarize_chunk(mock_agent_class, config: AppConfig, sample_chunks: list[CodeChunk], mock_run_result: Mock):
    """Test summarizing a single chunk"""
    # Setup the mock agent instance
    mock_agent = mock_agent_class.return_value
    mock_agent.run = AsyncMock(return_value=mock_run_result)

    summarizer = CodeSummarizer(config)
    result = await summarizer.summarize_chunk(sample_chunks[0])
    
    # Verify result
    assert isinstance(result, str)
    assert result == format_code_summary(sample_chunks[0].content, mock_run_result.data)
    
    # Verify agent was called with correct prompt
    call_args = mock_agent.run.call_args[0][0]
    assert "def hello()" in call_args
    assert "Says hello" in call_args

@patch('knowlang.summarizer.summarizer.Agent')
def test_chromadb_initialization(mock_agent_class, config: AppConfig):
    """Test ChromaDB initialization"""
    mock_agent = mock_agent_class.return_value
    
    summarizer = CodeSummarizer(config)
    assert summarizer.collection is not None
    
    # Verify we can create a new collection
    summarizer.db_client.delete_collection(config.db.collection_name)
    new_summarizer = CodeSummarizer(config)
    assert new_summarizer.collection is not None

@pytest.mark.asyncio
@patch('knowlang.summarizer.summarizer.generate_embedding')
@patch('knowlang.summarizer.summarizer.Agent')
async def test_process_and_store_chunk_with_embedding(
    mock_agent_class, 
    mock_embedding_generator, 
    config: AppConfig, 
    sample_chunks: list[CodeChunk], 
    mock_run_result: Mock
):
    """Test processing and storing a chunk with embedding"""
    # Setup the mock agent instance
    mock_agent = mock_agent_class.return_value
    mock_agent.run = AsyncMock(return_value=mock_run_result)
    
    # Setup mock embedding response
    mock_embedding = [0.1, 0.2, 0.3]  # Sample embedding vector
    mock_embedding_generator.return_value = mock_embedding
    
    summarizer = CodeSummarizer(config)
    
    # Mock the collection's add method
    summarizer.collection.add = Mock()
    
    # Process the chunk
    await summarizer.process_and_store_chunk(sample_chunks[0])

    code_summary = format_code_summary(sample_chunks[0].content, mock_run_result.data)
    
    # Verify ollama.embed was called with correct parameters
    mock_embedding_generator.assert_called_once_with(
        code_summary,
        config.embedding,
    )
    
    # Verify collection.add was called with correct parameters
    add_call = summarizer.collection.add.call_args
    assert add_call is not None
    
    kwargs = add_call[1]
    relative_path = Path(sample_chunks[0].file_path).relative_to(config.db.codebase_directory).as_posix()
    assert len(kwargs['embeddings']) == 3
    assert kwargs['embeddings'] == mock_embedding
    assert kwargs['documents'][0] == code_summary
    assert kwargs['ids'][0] == f"{relative_path}:{sample_chunks[0].start_line}-{sample_chunks[0].end_line}"
    
    # Verify metadata
    metadata = kwargs['metadatas'][0]
    assert metadata['file_path'] == relative_path, "File path must be relative"
    assert metadata['start_line'] == sample_chunks[0].start_line
    assert metadata['end_line'] == sample_chunks[0].end_line
    assert metadata['type'] == sample_chunks[0].type.value
    assert metadata['name'] == sample_chunks[0].name
    assert metadata['docstring'] == sample_chunks[0].docstring