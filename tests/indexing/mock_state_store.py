from pathlib import Path
from typing import Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from knowlang.indexing.state_store.base import FileState, FileChange


class MockStateStore(MagicMock):
    """Mock implementation of StateStore for testing using composition rather than inheritance."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Internal state for testing verification
        self.states: Dict[Path, FileState] = {}
        self.changes_to_return: List[FileChange] = []
        self.deleted_chunks: Set[str] = set()
        
        # Create AsyncMock methods that will properly return coroutines
        self.get_file_state = AsyncMock(side_effect=self._get_file_state)
        self.update_file_state = AsyncMock(side_effect=self._update_file_state)
        self.delete_file_state = AsyncMock(side_effect=self._delete_file_state)
        self.get_all_file_states = AsyncMock(side_effect=self._get_all_file_states)
        self.detect_changes = AsyncMock(side_effect=self._detect_changes)
    
    async def _get_file_state(self, file_path: Path) -> Optional[FileState]:
        """Mock implementation of get_file_state"""
        return self.states.get(file_path)
    
    async def _update_file_state(self, file_path: Path, chunk_ids: Set[str]) -> None:
        """Mock implementation of update_file_state matches the real signature"""
        # Get or create a FileState with the new chunk IDs
        if file_path in self.states:
            old_state = self.states[file_path]
            new_state = FileState(
                file_path=old_state.file_path,
                last_modified=old_state.last_modified,
                file_hash=old_state.file_hash,
                chunk_ids=chunk_ids
            )
        else:
            new_state = FileState(
                file_path=str(file_path),
                last_modified=datetime.now(),  # This would be set in the real implementation
                file_hash="mock_hash",
                chunk_ids=chunk_ids
            )
        self.states[file_path] = new_state
    
    async def _delete_file_state(self, file_path: Path) -> Set[str]:
        """Mock implementation of delete_file_state"""
        if file_path in self.states:
            chunks = self.states[file_path].chunk_ids
            self.deleted_chunks.update(chunks)
            del self.states[file_path]
            return chunks
        return set()
    
    async def _get_all_file_states(self) -> Dict[Path, FileState]:
        """Mock implementation of get_all_file_states"""
        return self.states
    
    async def _detect_changes(self, current_files: Set[Path]) -> List[FileChange]:
        """Return pre-configured changes for testing"""
        return self.changes_to_return

    def set_changes(self, changes: List[FileChange]) -> None:
        """Helper to set changes that will be returned by detect_changes"""
        self.changes_to_return = changes
        
    def set_file_state(self, file_path: Path, state: FileState) -> None:
        """Helper to directly set a file state for testing"""
        self.states[file_path] = state