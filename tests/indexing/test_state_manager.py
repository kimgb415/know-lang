from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from knowlang.configs import AppConfig
from knowlang.indexing.state_manager import StateManager
from knowlang.indexing.state_store.base import FileState
from knowlang.vector_stores.mock import MockVectorStore
from tests.indexing.mock_state_store import MockStateStore


@pytest.fixture
def mock_state_store():
    return MockStateStore()

@pytest.fixture
def mock_vector_store():
    return MockVectorStore()

@pytest.fixture
def mock_config():
    """Create a mock AppConfig that returns our mock stores"""
    return AppConfig()

@pytest.fixture
def state_manager(mock_config, mock_state_store, mock_vector_store):
    """Create StateManager with patched dependencies"""
    # Patch the state store creation
    with patch('knowlang.indexing.state_manager.get_state_store', return_value=mock_state_store):
        # Patch the vector store factory
        with patch('knowlang.indexing.state_manager.VectorStoreFactory.get', return_value=mock_vector_store):
            yield StateManager(mock_config)

def create_file_state(file_path: str, chunk_ids: set[str]) -> FileState:
    """Helper to create a test FileState"""
    return FileState(
        file_path=file_path,
        last_modified=datetime.now(),
        file_hash="test_hash",
        chunk_ids=chunk_ids
    )

@pytest.mark.asyncio
async def test_get_file_state(state_manager: StateManager, mock_state_store: MockStateStore):
    """Test retrieving a file state"""
    # Setup
    file_path = Path("test.py")
    test_state = create_file_state(str(file_path), {"chunk1", "chunk2"})
    mock_state_store.set_file_state(file_path, test_state)
    
    # Execute
    state = await state_manager.get_file_state(file_path)
    
    # Verify
    assert state == test_state
    mock_state_store.get_file_state.assert_called_once_with(file_path)

@pytest.mark.asyncio
async def test_update_file_state_new(state_manager: StateManager, mock_state_store: MockStateStore):
    """Test updating state for a new file"""
    # Setup
    file_path = Path("test.py")
    new_state = create_file_state(str(file_path), {"chunk1", "chunk2"})
    
    # Execute
    await state_manager.update_file_state(file_path, new_state)
    
    # Verify state was updated with the right chunks
    mock_state_store.update_file_state.assert_called_once_with(
        file_path, 
        new_state.chunk_ids
    )

@pytest.mark.asyncio
async def test_update_file_state_existing(state_manager: StateManager, mock_state_store: MockStateStore, mock_vector_store: MockVectorStore):
    """Test updating state for an existing file"""
    # Setup
    file_path = Path("test.py")
    old_state = create_file_state(str(file_path), {"old_chunk1", "old_chunk2"})
    new_state = create_file_state(str(file_path), {"new_chunk1"})
    
    # Add the old state to the mock store
    mock_state_store.set_file_state(file_path, old_state)
    
    # Execute
    await state_manager.update_file_state(file_path, new_state)
    
    # Verify the vector store delete was called with old chunks
    mock_vector_store.delete_mock.assert_called_once_with(list(old_state.chunk_ids))
    
    # Verify state store update was called with new chunks
    mock_state_store.update_file_state.assert_called_once_with(
        file_path, 
        new_state.chunk_ids
    )

@pytest.mark.asyncio
async def test_delete_file_state(state_manager: StateManager, mock_state_store: MockStateStore, mock_vector_store: MockVectorStore):
    """Test deleting a file state"""
    # Setup
    file_path = Path("test.py")
    test_state = create_file_state(str(file_path), {"chunk1", "chunk2"})
    mock_state_store.set_file_state(file_path, test_state)
    
    # Configure the mock to return the chunks when delete is called
    mock_state_store.delete_file_state.return_value = test_state.chunk_ids
    
    # Execute
    await state_manager.delete_file_state(file_path)
    
    # Verify state store delete was called
    mock_state_store.delete_file_state.assert_called_once_with(file_path)
    
    # Verify chunks were deleted from vector store
    mock_vector_store.delete_mock.assert_called_once_with(list(test_state.chunk_ids))