from pathlib import Path
from typing import Dict, List, Optional, Set

from knowlang.indexing.state_store.base import FileState, StateStore, FileChange

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