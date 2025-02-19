from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from dataclasses import dataclass

from knowlang.configs.config import DBConfig
from knowlang.indexing.state_store.base import FileChange, StateChangeType
from knowlang.indexing.codebase_manager import CodebaseManager
from knowlang.indexing.state_manager import StateManager
from knowlang.indexing.chunk_indexer import ChunkIndexer
from knowlang.core.types import CodeChunk
from knowlang.utils.fancy_log import FancyLogger

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
        codebase_manager: CodebaseManager,
        state_manager: StateManager,
        chunk_indexer: ChunkIndexer,
        db_config: DBConfig,
    ):
        self.codebase_manager = codebase_manager
        self.state_manager = state_manager
        self.chunk_indexer = chunk_indexer
        self.db_config = db_config

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
                # Handle deletions and modifications
                if change.change_type in (StateChangeType.MODIFIED, StateChangeType.DELETED):
                    old_state = await self.state_manager.get_file_state(change.path)
                    if old_state and old_state.chunk_ids:
                        stats.chunks_deleted += len(old_state.chunk_ids)
                        await self.state_manager.delete_file_state(change.path)
                
                # Handle additions and modifications
                if change.change_type in (StateChangeType.ADDED, StateChangeType.MODIFIED):
                    if change.path in chunks_by_file:
                        file_chunks = chunks_by_file[change.path]
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

    async def update_codebase(self, chunks: List[CodeChunk]) -> UpdateStats:
        """High-level method to update entire codebase incrementally"""
        try:
            # Get current files using codebase manager
            current_files = await self.codebase_manager.get_current_files(
                self.db_config.codebase_directory
            )
            
            # Detect changes using state manager
            changes = await self.state_manager.state_store.detect_changes(current_files)
            
            if not changes:
                LOG.info("No changes detected in codebase")
                return UpdateStats()
            
            LOG.info(f"Detected {len(changes)} changed files")
            
            # Process changes
            return await self.process_changes(changes, chunks)
            
        except Exception as e:
            LOG.error(f"Error updating codebase: {e}")
            stats = UpdateStats()
            stats.errors += 1
            return stats