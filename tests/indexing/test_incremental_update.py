import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock
from typing import Set

from knowlang.indexing.increment_update import IncrementalUpdater, UpdateStats
from knowlang.indexing.state_store.base import FileState, FileChange, StateChangeType
from knowlang.core.types import CodeChunk, CodeLocation, BaseChunkType, LanguageEnum
from knowlang.configs.config import DBConfig
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

def create_file_state(file_path: str, chunk_ids: Set[str]) -> FileState:
    """Helper to create a test FileState"""
    return FileState(
        file_path=file_path,
        last_modified=datetime.now(),
        file_hash="test_hash",
        chunk_ids=chunk_ids
    )

class MockCodebaseManager:
    async def get_current_files(self) -> Set[Path]:
        return {Path("test.py")}

    async def create_file_state(self, file_path: Path, chunk_ids: Set[str]) -> FileState:
        return create_file_state(str(file_path), chunk_ids)

class MockStateManager:
    def __init__(self, state_store):
        self.state_store = state_store
        self.get_file_state = AsyncMock()
        self.update_file_state = AsyncMock()
        self.delete_file_state = AsyncMock()

class MockChunkIndexer:
    def __init__(self):
        self.process_file_chunks = AsyncMock()

@pytest.fixture
def mock_state_store():
    return MockStateStore()

@pytest.fixture
def mock_state_manager(mock_state_store):
    return MockStateManager(mock_state_store)

@pytest.fixture
def mock_codebase_manager():
    return MockCodebaseManager()

@pytest.fixture
def mock_chunk_indexer():
    return MockChunkIndexer()

@pytest.fixture
def db_config():
    return DBConfig()

@pytest.fixture
def updater(mock_codebase_manager, mock_state_manager, mock_chunk_indexer, db_config):
    return IncrementalUpdater(
        codebase_manager=mock_codebase_manager,
        state_manager=mock_state_manager,
        chunk_indexer=mock_chunk_indexer,
        db_config=db_config
    )

@pytest.mark.asyncio
async def test_process_added_file(updater: IncrementalUpdater, mock_chunk_indexer: MockChunkIndexer):
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
async def test_process_modified_file(updater: IncrementalUpdater, mock_chunk_indexer: MockChunkIndexer):
    """Test processing a modified file"""
    # Setup
    file_path = Path("test.py")
    old_state = create_file_state("test.py", {"old_chunk"})
    updater.state_manager.get_file_state.return_value = old_state
    
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
    updater.state_manager.delete_file_state.assert_called_once()
    updater.state_manager.update_file_state.assert_called_once()

@pytest.mark.asyncio
async def test_process_deleted_file(updater: IncrementalUpdater):
    """Test processing a deleted file"""
    # Setup
    file_path = Path("test.py")
    old_state = create_file_state("test.py", {"chunk1", "chunk2"})
    updater.state_manager.get_file_state.return_value = old_state
    
    changes = [FileChange(path=file_path, change_type=StateChangeType.DELETED)]
    
    # Execute
    stats = await updater.process_changes(changes, [])
    
    # Verify
    assert stats.files_deleted == 1
    assert stats.chunks_deleted == 2
    assert stats.errors == 0
    updater.state_manager.delete_file_state.assert_called_once()

@pytest.mark.asyncio
async def test_process_error_handling(updater: IncrementalUpdater, mock_chunk_indexer: MockChunkIndexer):
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
async def test_update_codebase_no_changes(updater: IncrementalUpdater, mock_state_store: MockChunkIndexer):
    """Test update when no changes detected"""
    # Setup
    mock_state_store.set_changes([])
    
    # Execute
    stats = await updater.update_codebase([])
    
    # Verify
    assert stats.files_added == 0
    assert stats.files_modified == 0
    assert stats.files_deleted == 0
    assert stats.chunks_added == 0
    assert stats.chunks_deleted == 0
    assert stats.errors == 0