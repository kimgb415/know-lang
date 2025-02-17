from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Set, Dict
from pydantic import BaseModel


class StateChangeType(Enum(str)):
    """Types of changes in files"""
    ADDED = 'added'
    MODIFIED = 'modified'
    DELETED = 'deleted'

class FileState(BaseModel):
    """File state information"""
    file_path: str
    last_modified: datetime
    file_hash: str
    chunk_ids: Set[str]

class FileChange(BaseModel):
    """Represents a change in a file"""
    path: Path
    change_type: StateChangeType
    old_chunks: Set[str] = None

class StateStore(ABC):
    """Abstract base class for file state storage"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the state store"""
        pass
    
    @abstractmethod
    async def get_file_state(self, file_path: Path) -> Optional[FileState]:
        """Get current state of a file"""
        pass
    
    @abstractmethod
    async def update_file_state(self, file_path: Path, state: FileState) -> None:
        """Update or create file state"""
        pass
    
    @abstractmethod
    async def delete_file_state(self, file_path: Path) -> Set[str]:
        """Delete file state and return associated chunk IDs"""
        pass
    
    @abstractmethod
    async def get_all_file_states(self) -> Dict[Path, FileState]:
        """Get all file states"""
        pass

    @abstractmethod
    async def detect_changes(self, current_files: Set[Path]) -> List[FileChange]:
        """Detect changes in files since last update"""
        pass