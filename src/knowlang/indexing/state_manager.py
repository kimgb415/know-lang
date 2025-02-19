from typing import Set, Optional
from pathlib import Path

from knowlang.indexing.state_store.base import FileState, StateStore
from knowlang.vector_stores.base import VectorStore
from knowlang.utils.fancy_log import FancyLogger

LOG = FancyLogger(__name__)

class StateManager:
    """Manages file states and their associated chunks"""
    
    def __init__(self, state_store: StateStore, vector_store: VectorStore):
        self.state_store = state_store
        self.vector_store = vector_store

    async def get_file_state(self, file_path: Path) -> Optional[FileState]:
        """Get current state of a file"""
        return await self.state_store.get_file_state(file_path)

    async def update_file_state(self, file_path: Path, state: FileState) -> None:
        """Update file state and manage associated chunks"""
        # Get existing state to handle cleanup if needed
        old_state = await self.get_file_state(file_path)
        if old_state and old_state.chunk_ids:
            await self.vector_store.delete(list(old_state.chunk_ids))
            
        await self.state_store.update_file_state(file_path, state)

    async def delete_file_state(self, file_path: Path) -> None:
        """Delete file state and its chunks"""
        deleted_chunks = await self.state_store.delete_file_state(file_path)
        if deleted_chunks:
            await self.vector_store.delete(list(deleted_chunks))