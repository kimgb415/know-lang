from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from knowlang.configs.config import AppConfig
from knowlang.core.types import CodeChunk
from knowlang.indexing.chunk_indexer import ChunkIndexer
from knowlang.indexing.codebase_manager import CodebaseManager
from knowlang.indexing.state_manager import StateManager
from knowlang.indexing.state_store.base import FileChange, StateChangeType
from knowlang.utils import FancyLogger, convert_to_relative_path

LOG = FancyLogger(__name__)

@dataclass
class UpdateStats:
    """Statistics about the incremental update process"""
    files_added: int = 0
    files_modified: int = 0
    files_deleted: int = 0
    chunks_added: int = 0
    chunks_deleted: int = 0
    errors: int = 0

    def summary(self) -> str:
        """Get a human-readable summary of the update stats"""
        return (
            f"Update completed:\n"
            f"  Files: {self.files_added} added, {self.files_modified} modified, "
            f"{self.files_deleted} deleted\n"
            f"  Chunks: {self.chunks_added} added, {self.chunks_deleted} deleted\n"
            f"  Errors: {self.errors}"
        )

class IncrementalUpdater:
    """Orchestrates incremental updates to the vector store"""
    
    def __init__(
        self,
        app_config: AppConfig,
    ):
        self.app_config = app_config
        self.codebase_manager = CodebaseManager(app_config)
        self.state_manager = StateManager(app_config)
        self.chunk_indexer = ChunkIndexer(app_config)

    def _group_chunks_by_file(self, chunks: List[CodeChunk]) -> Dict[Path, List[CodeChunk]]:
        """Group chunks by their source file path"""
        chunks_by_file = defaultdict(list)
        for chunk in chunks:
            chunks_by_file[chunk.location.file_path].append(chunk)
        return dict(chunks_by_file)

    async def process_changes(
        self,
        changes: List[FileChange],
        chunks: List[CodeChunk]
    ) -> UpdateStats:
        """Process detected changes and update vector store"""
        stats = UpdateStats()
        chunks_by_file = self._group_chunks_by_file(chunks)
        
        for change in changes:
            try:
                # Handle deletions and modifications (remove old chunks)
                if change.change_type in (StateChangeType.MODIFIED, StateChangeType.DELETED):
                    old_state = await self.state_manager.get_file_state(change.path)
                    if old_state and old_state.chunk_ids:
                        stats.chunks_deleted += len(old_state.chunk_ids)
                        await self.state_manager.delete_file_state(change.path)
                
                # Handle additions and modifications (add new chunks)
                if change.change_type in (StateChangeType.ADDED, StateChangeType.MODIFIED):
                    change_path_str = convert_to_relative_path(change.path, self.app_config.db)
                    if change_path_str in chunks_by_file:
                        file_chunks = chunks_by_file[change_path_str]
                        chunk_ids = await self.chunk_indexer.process_file_chunks(
                            change.path, 
                            file_chunks
                        )
                        
                        if chunk_ids:
                            new_state = await self.codebase_manager.create_file_state(
                                change.path,
                                chunk_ids
                            )
                            await self.state_manager.update_file_state(
                                change.path,
                                new_state
                            )
                            stats.chunks_added += len(chunk_ids)
                
                # Update stats
                if change.change_type == StateChangeType.ADDED:
                    stats.files_added += 1
                elif change.change_type == StateChangeType.MODIFIED:
                    stats.files_modified += 1
                elif change.change_type == StateChangeType.DELETED:
                    stats.files_deleted += 1
                
            except Exception as e:
                LOG.error(f"Error processing change for {change.path}: {e}")
                stats.errors += 1
                continue
        
        LOG.info(stats.summary())
        return stats

    async def update_codebase(self, chunks: List[CodeChunk], file_changes: List[FileChange]) -> UpdateStats:
        """High-level method to update entire codebase incrementally"""
        try:
            if not file_changes:
                LOG.info("No changes detected in codebase")
                return UpdateStats()
            
            # Process changes
            return await self.process_changes(file_changes, chunks)
            
        except Exception as e:
            LOG.error(f"Error updating codebase: {e}")
            return UpdateStats(errors=1)