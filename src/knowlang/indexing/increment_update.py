# knowlang/core/incremental.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict

from knowlang.indexing.state_store.base import FileChange, StateStore, StateChangeType
from knowlang.core.types import CodeChunk
from knowlang.vector_stores.base import VectorStore
from knowlang.models.embeddings import generate_embedding
from knowlang.utils.fancy_log import FancyLogger
from knowlang.configs.config import AppConfig

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
    """Handles incremental updates to the vector store with improved error handling"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        state_store: StateStore,
        config: AppConfig
    ):
        self.vector_store = vector_store
        self.state_store = state_store
        self.config = config

    def _group_chunks_by_file(self, chunks: List[CodeChunk]) -> Dict[Path, List[CodeChunk]]:
        """Group chunks by their source file path"""
        chunks_by_file = defaultdict(list)
        for chunk in chunks:
            chunks_by_file[chunk.location.file_path].append(chunk)
        return dict(chunks_by_file)

    async def process_chunk(self, chunk: CodeChunk) -> str:
        """Process a single chunk and store in vector store"""
        try:
            # Generate summary using IndexingAgent (placeholder)
            summary = "Placeholder summary"  # This would be replaced with actual summary generation
            
            # Generate embedding
            embedding = generate_embedding(summary, self.config.embedding)
            
            # Create unique ID for chunk
            chunk_id = chunk.location.to_single_line()
            
            # Store in vector store
            await self.vector_store.add_documents(
                documents=[summary],
                embeddings=[embedding],
                metadatas=[chunk.model_dump()],
                ids=[chunk_id]
            )
            
            return chunk_id
            
        except Exception as e:
            LOG.error(f"Error processing chunk {chunk.location}: {e}")
            raise

    async def process_file_chunks(
        self, 
        file_path: Path,
        chunks: List[CodeChunk]
    ) -> List[str]:
        """Process all chunks from a single file"""
        chunk_ids = []
        for chunk in chunks:
            try:
                chunk_id = await self.process_chunk(chunk)
                chunk_ids.append(chunk_id)
            except Exception as e:
                LOG.error(f"Error processing chunk in {file_path}: {e}")
                # Continue processing other chunks
                continue
        return chunk_ids

    async def process_changes(
        self,
        changes: List[FileChange],
        chunks: List[CodeChunk]
    ) -> UpdateStats:
        """Process detected changes and update vector store with stats tracking"""
        stats = UpdateStats()
        chunks_by_file = self._group_chunks_by_file(chunks)
        
        for change in changes:
            try:
                # Handle deletions and modifications first
                if change.change_type in (StateChangeType.MODIFIED, StateChangeType.DELETED):
                    if change.old_chunks:
                        await self.vector_store.delete(list(change.old_chunks))
                        stats.chunks_deleted += len(change.old_chunks)
                
                # Handle additions and modifications
                if change.change_type in (StateChangeType.ADDED, StateChangeType.MODIFIED):
                    if change.path in chunks_by_file:
                        file_chunks = chunks_by_file[change.path]
                        chunk_ids = await self.process_file_chunks(
                            change.path, 
                            file_chunks
                        )
                        
                        if chunk_ids:  # Only update state if chunks were processed
                            await self.state_store.update_file_state(
                                change.path,
                                chunk_ids
                            )
                            stats.chunks_added += len(chunk_ids)
                
                # Update stats based on change type
                if change.change_type == StateChangeType.ADDED:
                    stats.files_added += 1
                elif change.change_type == StateChangeType.MODIFIED:
                    stats.files_modified += 1
                elif change.change_type == StateChangeType.DELETED:
                    # Remove file state for deleted files
                    await self.state_store.delete_file_state(change.path)
                    stats.files_deleted += 1
                
            except Exception as e:
                LOG.error(f"Error processing change for {change.path}: {e}")
                stats.errors += 1
                # Continue processing other changes
                continue
        
        LOG.info(stats.summary())
        return stats

    async def get_current_files(self, root_dir: Path) -> Set[Path]:
        """Get set of current files in directory with proper filtering"""
        current_files = set()
        
        try:
            for path in root_dir.rglob('*'):
                if path.is_file():
                    # Apply path pattern filtering from config
                    relative_path = path.relative_to(root_dir)
                    if self.config.parser.path_patterns.should_process_path(str(relative_path)):
                        current_files.add(path)
            
            return current_files
            
        except Exception as e:
            LOG.error(f"Error scanning directory {root_dir}: {e}")
            raise

    async def update_codebase(self, chunks: List[CodeChunk]) -> UpdateStats:
        """High-level method to update entire codebase incrementally"""
        try:
            # Get current files
            current_files = await self.get_current_files(
                self.config.db.codebase_directory
            )
            
            # Detect changes
            changes = await self.state_store.detect_changes(current_files)
            
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