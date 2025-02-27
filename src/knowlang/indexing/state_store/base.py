from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Type

from pydantic import BaseModel

from knowlang.configs import DBConfig


# ----------------- Data Models and Enums -----------------
class StateChangeType(str, Enum):
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

# ----------------- Base Interface & Registry -----------------

class StateStore(ABC):
    """Abstract base class for state storage implementations."""
    def __init__(self, config: DBConfig):
        self.config = config

    @abstractmethod
    async def get_file_state(self, file_path: Path) -> Optional[FileState]:
        pass

    @abstractmethod
    async def update_file_state(self, file_path: Path, chunk_ids: List[str]) -> None:
        pass

    @abstractmethod
    async def delete_file_state(self, file_path: Path) -> Set[str]:
        pass

    @abstractmethod
    async def get_all_file_states(self) -> Dict[Path, FileState]:
        pass

    @abstractmethod
    async def detect_changes(self, current_files: Set[Path]) -> List[FileChange]:
        pass

# Registry dictionary to map provider keys to implementations
STATE_STORE_REGISTRY: Dict[str, Type[StateStore]] = {}

def register_state_store(provider: str):
    """Decorator to register a state store implementation for a given provider key."""
    def decorator(cls: Type[StateStore]):
        STATE_STORE_REGISTRY[provider] = cls
        return cls
    return decorator

def get_state_store(config: DBConfig) -> StateStore:
    """Factory method to retrieve a state store instance based on configuration."""
    provider = config.state_store.provider
    if provider not in STATE_STORE_REGISTRY:
        raise ValueError(f"State store provider {provider} is not registered.")
    store_cls = STATE_STORE_REGISTRY[provider]
    return store_cls(config)