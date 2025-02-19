import pytest
from pathlib import Path
from datetime import datetime

from knowlang.indexing.state_manager import StateManager
from knowlang.indexing.state_store.base import FileState, StateStore
from knowlang.vector_stores.base import VectorStore
from knowlang.vector_stores.mock import MockVectorStore
from tests.indexing.mock_state_store import MockStateStore

@pytest.fixture
def mock_state_store():
    return MockStateStore()

@pytest.fixture
def mock_vector_store():
    return MockVectorStore()

@pytest.fixture
def state_manager(mock_state_store: StateStore, mock_vector_store: MockVectorStore):
    return StateManager(mock_state_store, mock_vector_store)

def create_file_state(file_path: str, chunk_ids: set[str]) -> FileState:
    """Helper to create a test FileState"""
    return FileState(
        file_path=file_path,
        last_modified=datetime.now(),
        file_hash="test_hash",
        chunk_ids=chunk_ids
    )

@pytest.mark.asyncio
async def test_get_file_state(state_manager: StateManager, mock_state_store: StateStore):
    """Test retrieving a file state"""
    # Setup
    file_path = Path("test.py")
    test_state = create_file_state(str(file_path), {"chunk1", "chunk2"})
    await mock_state_store.update_file_state(file_path, test_state)
    
    # Execute
    state = await state_manager.get_file_state(file_path)
    
    # Verify
    assert state == test_state

@pytest.mark.asyncio
async def test_update_file_state_new(state_manager: StateManager):
    """Test updating state for a new file"""
    # Setup
    file_path = Path("test.py")
    new_state = create_file_state(str(file_path), {"chunk1", "chunk2"})
    
    # Execute
    await state_manager.update_file_state(file_path, new_state)
    
    # Verify
    state = await state_manager.get_file_state(file_path)
    assert state == new_state

@pytest.mark.asyncio
async def test_update_file_state_existing(state_manager: StateManager, mock_vector_store: MockVectorStore):
    """Test updating state for an existing file"""
    # Setup
    file_path = Path("test.py")
    old_state = create_file_state(str(file_path), {"old_chunk1", "old_chunk2"})
    new_state = create_file_state(str(file_path), {"new_chunk1"})
    
    await state_manager.update_file_state(file_path, old_state)
    
    # Execute
    await state_manager.update_file_state(file_path, new_state)
    
    # Verify
    state = await state_manager.get_file_state(file_path)
    assert state == new_state
    
    # Verify old chunks were deleted
    assert "old_chunk1" not in mock_vector_store.documents
    assert "old_chunk2" not in mock_vector_store.documents

@pytest.mark.asyncio
async def test_delete_file_state(state_manager: StateManager, mock_vector_store: MockVectorStore):
    """Test deleting a file state"""
    # Setup
    file_path = Path("test.py")
    test_state = create_file_state(str(file_path), {"chunk1", "chunk2"})
    await state_manager.update_file_state(file_path, test_state)
    
    # Execute
    await state_manager.delete_file_state(file_path)
    
    # Verify state was removed
    state = await state_manager.get_file_state(file_path)
    assert state is None
    
    # Verify chunks were deleted
    assert "chunk1" not in mock_vector_store.documents
    assert "chunk2" not in mock_vector_store.documents