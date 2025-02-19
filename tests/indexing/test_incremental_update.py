import pytest
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
from unittest.mock import AsyncMock, patch

from knowlang.indexing.state_store.base import FileState, StateStore, FileChange, StateChangeType
from knowlang.indexing.increment_update import IncrementalUpdater
from knowlang.core.types import BaseChunkType, CodeChunk, CodeLocation, LanguageEnum
from knowlang.vector_stores.mock import MockVectorStore
from knowlang.configs.config import AppConfig

class MockStateStore(StateStore):
    """Mock implementation of StateStore for testing"""
    def __init__(self):
        self.states: Dict[Path, FileState] = {}
        self.changes_to_return: List[FileChange] = []
    
    async def initialize(self) -> None:
        pass
    
    async def get_file_state(self, file_path: Path) -> Optional[FileState]:
        return self.states.get(file_path)
    
    async def update_file_state(self, file_path: Path, state: FileState) -> None:
        self.states[file_path] = state
    
    async def delete_file_state(self, file_path: Path) -> Set[str]:
        if file_path in self.states:
            chunks = self.states[file_path].chunk_ids
            del self.states[file_path]
            return chunks
        return set()
    
    async def get_all_file_states(self) -> Dict[Path, FileState]:
        return self.states
    
    async def detect_changes(self, current_files: Set[Path]) -> List[FileChange]:
        return self.changes_to_return

    def set_changes(self, changes: List[FileChange]):
        """Helper to set changes that will be returned by detect_changes"""
        self.changes_to_return = changes

@pytest.fixture
def mock_config():
    return AppConfig()

@pytest.fixture
def mock_vector_store():
    return MockVectorStore()

@pytest.fixture
def mock_state_store():
    return MockStateStore()

@pytest.fixture
def updater(mock_config, mock_vector_store, mock_state_store):
    return IncrementalUpdater(
        vector_store=mock_vector_store,
        state_store=mock_state_store,
        config=mock_config
    )

def create_test_chunk(path: str, content: str) -> CodeChunk:
    """Helper to create test chunks"""
    location = CodeLocation(
        file_path=path,
        start_line=1,
        end_line=2
    )
    name = location.to_single_line()
    return CodeChunk(
        language=LanguageEnum.PYTHON,
        name=name,
        content=content,
        location=location,
        type=BaseChunkType.FUNCTION,
        metadata={}
    )

def create_test_file(tmp_path: Path, filename: str, content: str) -> Path:
    """Helper to create a test file with content"""
    file_path = tmp_path / filename
    file_path.write_text(content)
    return file_path

@pytest.mark.asyncio
async def test_process_new_file(updater: IncrementalUpdater, mock_state_store: MockStateStore, tmp_path: Path):
    """Test processing a newly added file"""
    # Setup - create actual file
    file_content = "def test(): pass\nclass Test: pass"
    file_path = create_test_file(tmp_path, "test.py", file_content)
    
    chunks = [
        create_test_chunk(str(file_path), "def test(): pass"),
        create_test_chunk(str(file_path), "class Test: pass")
    ]
    
    mock_state_store.set_changes([
        FileChange(path=file_path, change_type=StateChangeType.ADDED)
    ])
    
    # Execute
    stats = await updater.process_changes(mock_state_store.changes_to_return, chunks)
    
    # Verify
    assert stats.files_added == 1
    assert stats.chunks_added == 2
    assert stats.errors == 0
    
    # Verify state was updated with correct file information
    state = await mock_state_store.get_file_state(file_path)
    assert state is not None
    assert len(state.chunk_ids) == 2
    assert state.file_path == str(file_path)
    assert isinstance(state.last_modified, datetime)
    assert state.file_hash  # Verify hash was computed

@pytest.mark.asyncio
async def test_process_modified_file(updater: IncrementalUpdater, mock_state_store: MockStateStore, tmp_path: Path):
    """Test processing a modified file"""
    # Setup - create and modify file
    file_path = create_test_file(tmp_path, "test.py", "def test_modified(): pass")
    
    # Create initial state
    initial_state = FileState(
        file_path=str(file_path),
        last_modified=datetime.fromtimestamp(file_path.stat().st_mtime),
        file_hash="old_hash",
        chunk_ids={"chunk1", "chunk2"}
    )
    await mock_state_store.update_file_state(file_path, initial_state)
    
    # Add new content to simulate modification
    new_chunks = [
        create_test_chunk(str(file_path), "def test_modified(): pass")
    ]
    
    mock_state_store.set_changes([
        FileChange(
            path=file_path,
            change_type=StateChangeType.MODIFIED,
            old_chunks={"chunk1", "chunk2"}
        )
    ])
    
    # Execute
    stats = await updater.process_changes(mock_state_store.changes_to_return, new_chunks)
    
    # Verify
    assert stats.files_modified == 1
    assert stats.chunks_deleted == 2
    assert stats.chunks_added == 1
    assert stats.errors == 0
    
    # Verify updated state
    new_state = await mock_state_store.get_file_state(file_path)
    assert new_state is not None
    assert new_state.file_hash != initial_state.file_hash
    assert len(new_state.chunk_ids) == 1

@pytest.mark.asyncio
async def test_process_deleted_file(updater: IncrementalUpdater, mock_state_store: MockStateStore, tmp_path: Path):
    """Test processing a deleted file"""
    # Setup - create file then delete it
    file_path = create_test_file(tmp_path, "test.py", "")
    
    # Create initial state
    initial_state = FileState(
        file_path=str(file_path),
        last_modified=datetime.fromtimestamp(file_path.stat().st_mtime),
        file_hash="hash",
        chunk_ids={"chunk1", "chunk2"}
    )
    await mock_state_store.update_file_state(file_path, initial_state)
    
    # Delete the file
    file_path.unlink()
    
    mock_state_store.set_changes([
        FileChange(
            path=file_path,
            change_type=StateChangeType.DELETED,
            old_chunks={"chunk1", "chunk2"}
        )
    ])
    
    # Execute
    stats = await updater.process_changes(mock_state_store.changes_to_return, [])
    
    # Verify
    assert stats.files_deleted == 1
    assert stats.chunks_deleted == 2
    assert stats.errors == 0
    
    # Verify state was removed
    assert await mock_state_store.get_file_state(file_path) is None

@pytest.mark.asyncio
async def test_error_handling(updater: IncrementalUpdater, mock_state_store: MockStateStore, tmp_path: Path):
    """Test error handling during processing"""
    # Setup - create file
    file_path = create_test_file(tmp_path, "test.py", "def test(): pass")
    chunks = [
        create_test_chunk(str(file_path), "def test(): pass")
    ]
    
    # Simulate vector store error
    updater.vector_store.add_documents = AsyncMock(side_effect=Exception("Test error"))
    
    mock_state_store.set_changes([
        FileChange(path=file_path, change_type=StateChangeType.ADDED)
    ])
    
    # Execute
    stats = await updater.process_changes(mock_state_store.changes_to_return, chunks)
    
    # Verify
    assert stats.errors == 1
    assert stats.files_added == 1
    assert stats.chunks_added == 0
    
    # Verify no state was created due to error
    assert await mock_state_store.get_file_state(file_path) is None

@pytest.mark.asyncio
async def test_no_changes(updater: IncrementalUpdater, mock_state_store: MockStateStore):
    """Test when no changes are detected"""
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

@pytest.mark.asyncio
async def test_get_current_files(updater: IncrementalUpdater, tmp_path: Path):
    """Test file scanning with pattern filtering"""
    # Create test files with content
    create_test_file(tmp_path, "test.py", "def test(): pass")
    create_test_file(tmp_path, "ignored.pyc", "ignored content")
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    create_test_file(subdir, "test2.py", "def test2(): pass")
    
    updater.config.db.codebase_directory = tmp_path
    
    # Execute
    current_files = await updater.get_current_files(tmp_path)
    
    # Verify - should only include .py files
    assert len(current_files) == 2
    assert tmp_path / "test.py" in current_files
    assert tmp_path / "subdir" / "test2.py" in current_files