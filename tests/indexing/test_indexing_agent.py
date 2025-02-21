import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from knowlang.configs.config import AppConfig
from knowlang.core.types import (BaseChunkType, CodeChunk, CodeLocation,
                                 LanguageEnum)
from knowlang.indexing.indexing_agent import IndexingAgent
from knowlang.utils import format_code_summary
from knowlang.vector_stores.base import VectorStoreError
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

@pytest.mark.asyncio
@patch('knowlang.indexing.indexing_agent.generate_embedding')
@patch('knowlang.indexing.indexing_agent.Agent')
async def test_process_and_store_chunk(
    mock_agent_class,
    mock_embedding_generator,
    config: AppConfig,
    sample_chunks: list[CodeChunk],
    mock_run_result: Mock,
    mock_vector_store: MockVectorStore,
    indexing_agent: IndexingAgent
):
    """Test processing and storing a chunk with embedding"""
    # Setup the mock agent instance
    mock_agent = mock_agent_class.return_value
    mock_agent.run = AsyncMock(return_value=mock_run_result)
    
    # Setup mock embedding response
    mock_embedding = [[0.1, 0.2, 0.3]]  # Sample embedding vector
    mock_embedding_generator.return_value = mock_embedding
    
    indexing_agent.agent = mock_agent
    await indexing_agent.process_and_store_chunk(sample_chunks[0])

    # Verify the document was added to the store
    assert len(mock_vector_store.documents) == 1
    doc_id = sample_chunks[0].location.to_single_line()
    stored_doc = mock_vector_store.documents[doc_id]
    
    # Verify document content
    expected_summary = format_code_summary(sample_chunks[0].content, mock_run_result.data)
    assert stored_doc == expected_summary
    
    # Verify metadata
    metadata = mock_vector_store.metadata[doc_id]
    assert metadata['file_path'] == sample_chunks[0].location.file_path
    assert metadata['start_line'] == sample_chunks[0].location.start_line
    assert metadata['end_line'] == sample_chunks[0].location.end_line
    assert metadata['type'] == sample_chunks[0].type.value
    assert metadata['name'] == sample_chunks[0].name

    # Verify embedding
    assert mock_vector_store.embeddings[doc_id] == mock_embedding[0]

@pytest.mark.asyncio
@patch('knowlang.indexing.indexing_agent.generate_embedding')
@patch('knowlang.indexing.indexing_agent.Agent')
async def test_process_chunks_error_handling(
    mock_agent_class,
    mock_embedding_generator,
    config: AppConfig,
    sample_chunks: list[CodeChunk],
    mock_vector_store: MockVectorStore,
    mock_run_result: Mock,
    indexing_agent: IndexingAgent
):
    """Test error handling during chunk processing"""
    mock_agent = mock_agent_class.return_value
    mock_agent.run = AsyncMock(return_value=mock_run_result)
    indexing_agent.agent = mock_agent

    # Setup mock embedding response
    mock_embedding = [[0.1, 0.2, 0.3]]  # Sample embedding vector
    mock_embedding_generator.return_value = mock_embedding

    # Setup mock to fail on the first chunk
    mock_vector_store.add_error = VectorStoreError("Test error")
    
    # Process should continue despite errors
    await indexing_agent.process_chunks(sample_chunks)
    
    # Reset mock store for second test
    mock_vector_store.reset()
    mock_vector_store.add_error = None
    
    # Test partial success
    mock_agent = mock_agent_class.return_value
    mock_agent.run = AsyncMock(side_effect=[
        Exception("First chunk fails"),
        Mock(data="Second chunk works")
    ])
    
    await indexing_agent.process_chunks(sample_chunks)
    
    # Should have processed the second chunk successfully
    assert len(mock_vector_store.documents) == 1