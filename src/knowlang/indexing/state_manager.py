import logging
from pathlib import Path
from typing import Optional

from knowlang.configs import AppConfig
from knowlang.indexing.state_store.base import FileState, get_state_store
from knowlang.vector_stores.factory import VectorStoreFactory

LOG = logging.getLogger(__name__)

class StateManager:
    """Manages file states and their associated chunks"""
    
    def __init__(self, config: AppConfig):
        self.state_store = get_state_store(config.db)
        self.vector_store = VectorStoreFactory.get(config.db, config.embedding)

    async def get_file_state(self, file_path: Path) -> Optional[FileState]:
        """Get current state of a file"""
        return await self.state_store.get_file_state(file_path)

    async def update_file_state(self, file_path: Path, state: FileState) -> None:
        """Update file state and manage associated chunks"""
        # Get existing state to handle cleanup if needed
        old_state = await self.get_file_state(file_path)
        if old_state and old_state.chunk_ids:
            await self.vector_store.delete(list(old_state.chunk_ids))
            
        await self.state_store.update_file_state(file_path, state.chunk_ids)

    async def delete_file_state(self, file_path: Path) -> None:
        """Delete file state and its chunks"""
        deleted_chunks = await self.state_store.delete_file_state(file_path)
        if deleted_chunks:
            await self.vector_store.delete(list(deleted_chunks))