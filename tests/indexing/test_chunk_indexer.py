from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from knowlang.configs import AppConfig
from knowlang.core.types import (BaseChunkType, CodeChunk, CodeLocation,
                                 LanguageEnum)
from knowlang.indexing.chunk_indexer import ChunkIndexer
from knowlang.indexing.indexing_agent import IndexingAgent
from knowlang.vector_stores.mock import MockVectorStore


def create_test_chunk(file_path: str, content: str, start_line=1, end_line=2) -> CodeChunk:
    """Helper to create test chunks"""
    location = CodeLocation(
        file_path=file_path,
        start_line=start_line,
        end_line=end_line
    )
    return CodeChunk(
        language=LanguageEnum.PYTHON,
        name=location.to_single_line(),
        content=content,
        location=location,
        type=BaseChunkType.FUNCTION,
        metadata={}
    )

@pytest.fixture
def mock_config():
    return AppConfig()

@pytest.fixture
def mock_vector_store():
    return MockVectorStore()

@pytest.fixture
def mock_indexing_agent():
    with patch('knowlang.indexing.chunk_indexer.IndexingAgent') as mock_agent_cls:
        with patch('knowlang.indexing.chunk_indexer.generate_embedding', return_value=[0.1, 0.2, 0.3]) as mock_generate_embedding:
            mock_agent = Mock()
            mock_agent.summarize_chunk = AsyncMock(return_value="Test summary")
            mock_agent_cls.return_value = mock_agent
            yield mock_agent

@pytest.fixture
def chunk_indexer(mock_config, mock_vector_store, mock_indexing_agent):
    with patch('knowlang.indexing.chunk_indexer.VectorStoreFactory.get', return_value=mock_vector_store):
        indexer = ChunkIndexer(mock_config)
        indexer.indexing_agent = mock_indexing_agent
        
        yield indexer

@pytest.mark.asyncio
async def test_process_single_chunk(chunk_indexer: ChunkIndexer, mock_indexing_agent: IndexingAgent):
    """Test processing a single code chunk"""
    # Create test chunk
    chunk = create_test_chunk("test.py", "def test(): pass")
    
    # Process chunk
    chunk_id = await chunk_indexer.process_chunk(chunk)
    
    # Verify
    assert chunk_id == chunk.location.to_single_line()
    
    # Verify document was added to vector store
    docs = await chunk_indexer.vector_store.get_all()
    assert len(docs) == 1
    assert docs[0].document == "def test(): pass"

@pytest.mark.asyncio
async def test_process_multiple_chunks(chunk_indexer: ChunkIndexer, mock_indexing_agent: IndexingAgent):
    """Test processing multiple chunks from a file"""
    # Create test chunks
    chunks = [
        create_test_chunk("test.py", "def test1(): pass", start_line=1, end_line=2),
        create_test_chunk("test.py", "def test2(): pass", start_line=10, end_line=20),
        create_test_chunk("test.py", "class Test: pass", start_line=100, end_line=200)
    ]
    
    # Process chunks
    chunk_ids = await chunk_indexer.process_file_chunks(Path("test.py"), chunks)
    
    # Verify
    assert len(chunk_ids) == 3
    assert mock_indexing_agent.summarize_chunk.call_count == 0
    
    # Verify all documents were added to vector store
    docs = await chunk_indexer.vector_store.get_all()
    assert len(docs) == 3

@pytest.mark.asyncio
async def test_process_multiple_chunks_with_summarization(chunk_indexer: ChunkIndexer, mock_indexing_agent: IndexingAgent):
    """Test processing multiple chunks from a file"""
    # Enable code summarization
    chunk_indexer.config.parser.enable_code_summarization = True


    # Create test chunks
    chunks = [
        create_test_chunk("test.py", "def test1(): pass", start_line=1, end_line=2),
        create_test_chunk("test.py", "def test2(): pass", start_line=10, end_line=20),
        create_test_chunk("test.py", "class Test: pass", start_line=100, end_line=200)
    ]
    
    # Process chunks
    chunk_ids = await chunk_indexer.process_file_chunks(Path("test.py"), chunks)
    
    # Verify
    assert len(chunk_ids) == 3
    assert mock_indexing_agent.summarize_chunk.call_count == 3
    
    # Verify all documents were added to vector store
    docs = await chunk_indexer.vector_store.get_all()
    assert len(docs) == 3

@pytest.mark.asyncio
async def test_error_handling_during_summarization(chunk_indexer: ChunkIndexer, mock_indexing_agent: IndexingAgent):
    """Test error handling during chunk processing"""
    # Enable code summarization
    chunk_indexer.config.parser.enable_code_summarization = True

    # Configure mock to raise exception
    mock_indexing_agent.summarize_chunk.side_effect = Exception("Test error")
    
    # Create test chunks
    chunks = [
        create_test_chunk("test.py", "def test1(): pass"),
        create_test_chunk("test.py", "def test2(): pass")
    ]
    
    # Process chunks - should continue despite errors
    chunk_ids = await chunk_indexer.process_file_chunks(Path("test.py"), chunks)
    
    # Verify no chunks were processed successfully
    assert len(chunk_ids) == 0
    
    # Verify error didn't prevent trying to process all chunks
    assert mock_indexing_agent.summarize_chunk.call_count == 2