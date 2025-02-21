import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, AsyncMock, MagicMock

from knowlang.indexing.chunk_indexer import ChunkIndexer
from knowlang.indexing.increment_update import IncrementalUpdater, UpdateStats
from knowlang.indexing.state_manager import StateManager
from knowlang.indexing.state_store.base import FileState, FileChange, StateChangeType
from knowlang.core.types import CodeChunk, CodeLocation, BaseChunkType, LanguageEnum
from knowlang.configs.config import AppConfig, DBConfig
from tests.indexing.mock_state_store import MockStateStore
from knowlang.vector_stores.mock import MockVectorStore

def create_test_chunk(file_path: str, content: str) -> CodeChunk:
    """Helper to create test chunks"""
    location = CodeLocation(
        file_path=file_path,
        start_line=1,
        end_line=2
    )
    return CodeChunk(
        language=LanguageEnum.PYTHON,
        name=location.to_single_line(),
        content=content,
        location=location,
        type=BaseChunkType.FUNCTION,
        metadata={}
    )

def create_file_state(file_path: str, chunk_ids: set[str]) -> FileState:
    """Helper to create a test FileState"""
    return FileState(
        file_path=file_path,
        last_modified=datetime.now(),
        file_hash="test_hash",
        chunk_ids=chunk_ids
    )

@pytest.fixture
def mock_app_config():
    """Create a mock AppConfig"""
    config = AppConfig()
    config.db = DBConfig()
    return config

@pytest.fixture
def mock_codebase_manager():
    """Create a mock CodebaseManager"""
    codebase_manager = AsyncMock()
    codebase_manager.create_file_state = AsyncMock()
    codebase_manager.create_file_state.side_effect = lambda file_path, chunk_ids: create_file_state(str(file_path), set(chunk_ids))
    return codebase_manager

@pytest.fixture
def mock_state_manager():
    """Create a mock StateManager"""
    state_manager = AsyncMock()
    state_manager.get_file_state = AsyncMock()
    state_manager.update_file_state = AsyncMock()
    state_manager.delete_file_state = AsyncMock()
    return state_manager

@pytest.fixture
def mock_chunk_indexer():
    """Create a mock ChunkIndexer"""
    chunk_indexer = AsyncMock()
    chunk_indexer.process_file_chunks = AsyncMock()
    return chunk_indexer

@pytest.fixture
def updater(mock_app_config, mock_codebase_manager, mock_state_manager, mock_chunk_indexer):
    """Create an IncrementalUpdater with mocked dependencies"""
    # Create patch for direct initialization
    with patch('knowlang.indexing.increment_update.CodebaseManager', return_value=mock_codebase_manager):
        with patch('knowlang.indexing.increment_update.StateManager', return_value=mock_state_manager):
            with patch('knowlang.indexing.increment_update.ChunkIndexer', return_value=mock_chunk_indexer):
                updater = IncrementalUpdater(app_config=mock_app_config)
                # Replace the mocks directly to ensure proper patching
                updater.codebase_manager = mock_codebase_manager
                updater.state_manager = mock_state_manager
                updater.chunk_indexer = mock_chunk_indexer
                yield updater

@pytest.mark.asyncio
async def test_process_added_file(updater: IncrementalUpdater, mock_chunk_indexer: ChunkIndexer):
    """Test processing a newly added file"""
    # Setup
    file_path = Path("test.py")
    chunks = [create_test_chunk("test.py", "def test(): pass")]
    mock_chunk_indexer.process_file_chunks.return_value = {"chunk1"}
    
    changes = [FileChange(path=file_path, change_type=StateChangeType.ADDED)]
    
    # Execute
    stats = await updater.process_changes(changes, chunks)
    
    # Verify
    assert stats.files_added == 1
    assert stats.chunks_added == 1
    assert stats.errors == 0
    mock_chunk_indexer.process_file_chunks.assert_called_once()
    updater.state_manager.update_file_state.assert_called_once()

@pytest.mark.asyncio
async def test_process_modified_file(updater: IncrementalUpdater, mock_chunk_indexer: ChunkIndexer, mock_state_manager: StateManager):
    """Test processing a modified file"""
    # Setup
    file_path = Path("test.py")
    old_state = create_file_state("test.py", {"old_chunk"})
    mock_state_manager.get_file_state.return_value = old_state
    
    chunks = [create_test_chunk("test.py", "def test_modified(): pass")]
    mock_chunk_indexer.process_file_chunks.return_value = {"new_chunk"}
    
    changes = [FileChange(path=file_path, change_type=StateChangeType.MODIFIED)]
    
    # Execute
    stats = await updater.process_changes(changes, chunks)
    
    # Verify
    assert stats.files_modified == 1
    assert stats.chunks_deleted == 1
    assert stats.chunks_added == 1
    assert stats.errors == 0
    
    # Verify method calls
    mock_state_manager.get_file_state.assert_called_with(file_path)
    mock_state_manager.delete_file_state.assert_called_once_with(file_path)
    mock_chunk_indexer.process_file_chunks.assert_called_once()
    mock_state_manager.update_file_state.assert_called_once()

@pytest.mark.asyncio
async def test_process_deleted_file(updater: IncrementalUpdater, mock_state_manager: StateManager):
    """Test processing a deleted file"""
    # Setup
    file_path = Path("test.py")
    old_state = create_file_state("test.py", {"chunk1", "chunk2"})
    mock_state_manager.get_file_state.return_value = old_state
    
    changes = [FileChange(path=file_path, change_type=StateChangeType.DELETED)]
    
    # Execute
    stats = await updater.process_changes(changes, [])
    
    # Verify
    assert stats.files_deleted == 1
    assert stats.chunks_deleted == 2
    assert stats.errors == 0
    
    # Verify method calls
    mock_state_manager.get_file_state.assert_called_with(file_path)
    mock_state_manager.delete_file_state.assert_called_once_with(file_path)

@pytest.mark.asyncio
async def test_process_error_handling(updater: IncrementalUpdater, mock_chunk_indexer: ChunkIndexer):
    """Test error handling during processing"""
    # Setup
    file_path = Path("test.py")
    chunks = [create_test_chunk("test.py", "def test(): pass")]
    mock_chunk_indexer.process_file_chunks.side_effect = Exception("Test error")
    
    changes = [FileChange(path=file_path, change_type=StateChangeType.ADDED)]
    
    # Execute
    stats = await updater.process_changes(changes, chunks)
    
    # Verify
    assert stats.errors == 1
    assert stats.files_added == 0
    assert stats.chunks_added == 0

@pytest.mark.asyncio
async def test_update_codebase_no_changes(updater: IncrementalUpdater):
    """Test update when no changes detected"""
    # Execute
    stats = await updater.update_codebase([], [])
    
    # Verify
    assert stats.files_added == 0
    assert stats.files_modified == 0
    assert stats.files_deleted == 0
    assert stats.chunks_added == 0
    assert stats.chunks_deleted == 0
    assert stats.errors == 0

@pytest.mark.asyncio
async def test_update_codebase_with_changes(updater: IncrementalUpdater):
    """Test update with detected changes"""
    # Setup
    file_path = Path("test.py")
    chunks = [create_test_chunk("test.py", "def test(): pass")]
    changes = [FileChange(path=file_path, change_type=StateChangeType.ADDED)]
    
    # Create a spy on process_changes
    original_process_changes = updater.process_changes
    updater.process_changes = AsyncMock()
    updater.process_changes.return_value = UpdateStats(files_added=1, chunks_added=1)
    
    # Execute
    stats = await updater.update_codebase(chunks, changes)
    
    # Verify
    assert stats.files_added == 1
    assert stats.chunks_added == 1
    
    # Verify process_changes was called with the right arguments
    updater.process_changes.assert_called_once_with(changes, chunks)
    
    # Restore the original method
    updater.process_changes = original_process_changes
    
@pytest.mark.asyncio
async def test_update_codebase_exception(updater: IncrementalUpdater):
    """Test handling of exceptions during update"""
    # Setup
    file_path = Path("test.py")
    chunks = [create_test_chunk("test.py", "def test(): pass")]
    changes = [FileChange(path=file_path, change_type=StateChangeType.ADDED)]
    
    # Create a spy that raises an exception
    updater.process_changes = AsyncMock(side_effect=Exception("Update failed"))
    
    # Execute
    stats = await updater.update_codebase(chunks, changes)
    
    # Verify
    assert stats.errors == 1
    assert stats.files_added == 0
    assert stats.chunks_added == 0