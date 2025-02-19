from pathlib import Path
import hashlib
from datetime import datetime
from typing import Set

from knowlang.indexing.state_store.base import FileState
from knowlang.utils.chunking_util import convert_to_relative_path
from knowlang.configs.config import AppConfig
from knowlang.utils.fancy_log import FancyLogger

LOG = FancyLogger(__name__)

class CodebaseManager:
    """Manages file-level operations and state creation"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        
    async def get_current_files(self) -> Set[Path]:
        """Get set of current files in directory with proper filtering"""
        current_files = set()
        
        try:
            for path in self.config.db.codebase_directory.rglob('*'):
                if path.is_file():
                    relative_path = convert_to_relative_path(path, self.config.db)
                    if self.config.parser.path_patterns.should_process_path(relative_path):
                        current_files.add(path)
            
            return current_files
            
        except Exception as e:
            LOG.error(f"Error scanning directory {self.config.db.codebase_directory}: {e}")
            raise

    async def compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file contents"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    async def create_file_state(self, file_path: Path, chunk_ids: Set[str]) -> FileState:
        """Create a new FileState object for a file"""
        return FileState(
            file_path=str(file_path),
            last_modified=datetime.fromtimestamp(file_path.stat().st_mtime),
            file_hash=await self.compute_file_hash(file_path),
            chunk_ids=chunk_ids
        )