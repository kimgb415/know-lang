from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict
import hashlib
from datetime import datetime

from knowlang.indexing.state_store.base import FileChange, StateStore, StateChangeType, FileState
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

    async def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file contents"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    async def _create_file_state(
        self, 
        file_path: Path, 
        chunk_ids: Set[str]
    ) -> FileState:
        """Create a new FileState object for a file"""
        return FileState(
            file_path=str(file_path),
            last_modified=datetime.fromtimestamp(file_path.stat().st_mtime),
            file_hash=await self._compute_file_hash(file_path),
            chunk_ids=chunk_ids
        )

    async def process_chunk(self, chunk: CodeChunk) -> str:
        """Process a single chunk and store in vector store"""
        try:
            # Generate summary using IndexingAgent (placeholder)
            # TODO: replace with actual summary generation
            summary = "Placeholder summary"  
            
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
    ) -> Set[str]:
        """Process all chunks from a single file and return set of chunk IDs"""
        chunk_ids = set()
        for chunk in chunks:
            try:
                chunk_id = await self.process_chunk(chunk)
                chunk_ids.add(chunk_id)
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
                # Handle deletions and modifications first - remove old chunks
                if change.change_type in (StateChangeType.MODIFIED, StateChangeType.DELETED):
                    old_state = await self.state_store.get_file_state(change.path)
                    if old_state and old_state.chunk_ids:
                        await self.vector_store.delete(list(old_state.chunk_ids))
                        stats.chunks_deleted += len(old_state.chunk_ids)
                
                # Handle additions and modifications - add new chunks
                if change.change_type in (StateChangeType.ADDED, StateChangeType.MODIFIED):
                    # convert the Path to str for comparison
                    change_path_str = str(change.path)
                    if change_path_str in chunks_by_file:
                        file_chunks = chunks_by_file[change_path_str]
                        chunk_ids = await self.process_file_chunks(
                            change.path, 
                            file_chunks
                        )
                        
                        if chunk_ids:  # Only update state if chunks were processed
                            # Create new file state
                            new_state = await self._create_file_state(
                                change.path,
                                chunk_ids
                            )
                            await self.state_store.update_file_state(
                                change.path,
                                new_state
                            )
                            stats.chunks_added += len(chunk_ids)
                
                # Update stats and handle file state based on change type
                if change.change_type == StateChangeType.ADDED:
                    stats.files_added += 1
                elif change.change_type == StateChangeType.MODIFIED:
                    stats.files_modified += 1
                elif change.change_type == StateChangeType.DELETED:
                    # Remove file state for deleted files
                    deleted_chunks = await self.state_store.delete_file_state(change.path)
                    if deleted_chunks:
                        await self.vector_store.delete(list(deleted_chunks))
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