# knowlang/core/state_store/sqlite.py
from __future__ import annotations
from datetime import datetime
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set
from sqlalchemy import create_engine, Column, String, DateTime, Integer, ForeignKey, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session, sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from knowlang.configs.state_store_config import StateStoreConfig
from knowlang.core.types import StateStoreProvider
from knowlang.indexing.state_store.base import (
    StateChangeType, 
    FileChange, 
    StateStore, 
    FileState as FileStateBase
)
from knowlang.utils.fancy_log import FancyLogger

LOG = FancyLogger(__name__)
Base = declarative_base()

class FileStateModel(Base):
    """SQLAlchemy model for file states"""
    __tablename__ = 'file_states'
    
    id = Column(Integer, primary_key=True)
    file_path = Column(String, unique=True, index=True)
    last_modified = Column(DateTime)
    file_hash = Column(String)
    chunks = relationship(
        "ChunkStateModel", 
        back_populates="file", 
        cascade="all, delete-orphan"
    )

    @classmethod
    def from_file(cls, file_path: Path, file_hash: str) -> FileStateModel:
        """Create FileStateModel from a file path"""
        return cls(
            file_path=str(file_path),
            last_modified=datetime.fromtimestamp(file_path.stat().st_mtime),
            file_hash=file_hash
        )

class ChunkStateModel(Base):
    """SQLAlchemy model for chunk states"""
    __tablename__ = 'chunk_states'
    
    id = Column(Integer, primary_key=True)
    chunk_id = Column(String, unique=True, index=True)
    file_id = Column(Integer, ForeignKey('file_states.id'))
    file = relationship("FileStateModel", back_populates="chunks")

class SQLiteStateStore(StateStore):
    """SQLite implementation of state storage using SQLAlchemy"""
    
    def __init__(self, config: StateStoreConfig):
        """Initialize SQLite database with configuration"""
        if config.type != StateStoreProvider.SQLITE:
            raise ValueError(f"Invalid store type for SQLiteStateStore: {config.type}")
            
        self.config = config
        connection_args = config.get_connection_args()
        
        self.engine = create_engine(
            connection_args.pop('url'),
            **connection_args
        )
        self.Session = sessionmaker(bind=self.engine)
        LOG.debug(f"Initialized SQLite state store at {config.store_path}")
        
    async def initialize(self) -> None:
        """Initialize database schema"""
        Base.metadata.create_all(self.engine)
        LOG.info(f"Initialized SQLite state store schema at {self.config.store_path}")

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file contents"""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except IOError as e:
            LOG.error(f"Error computing hash for {file_path}: {e}")
            raise

    async def get_file_state(self, file_path: Path) -> Optional[FileStateBase]:
        """Get current state of a file"""
        try:
            with Session(self.engine) as session:
                stmt = select(FileStateModel).where(
                    FileStateModel.file_path == str(file_path)
                )
                result = session.execute(stmt).scalar_one_or_none()
                
                if result:
                    return FileStateBase(
                        file_path=str(result.file_path),
                        last_modified=result.last_modified,
                        file_hash=result.file_hash,
                        chunk_ids={chunk.chunk_id for chunk in result.chunks}
                    )
                return None
        except SQLAlchemyError as e:
            LOG.error(f"Database error getting file state for {file_path}: {e}")
            raise

    async def update_file_state(
        self, 
        file_path: Path, 
        chunk_ids: List[str]
    ) -> None:
        """Update or create file state"""
        try:
            with Session(self.engine) as session:
                # Compute new file hash
                file_hash = self._compute_file_hash(file_path)
                
                # Get or create file state
                file_state = session.execute(
                    select(FileStateModel).where(
                        FileStateModel.file_path == str(file_path)
                    )
                ).scalar_one_or_none()
                
                if not file_state:
                    file_state = FileStateModel.from_file(file_path, file_hash)
                    session.add(file_state)
                else:
                    file_state.last_modified = datetime.fromtimestamp(
                        file_path.stat().st_mtime
                    )
                    file_state.file_hash = file_hash
                
                # Update chunks
                session.query(ChunkStateModel).filter_by(
                    file_id=file_state.id
                ).delete()
                
                for chunk_id in chunk_ids:
                    chunk_state = ChunkStateModel(
                        chunk_id=chunk_id,
                        file=file_state
                    )
                    session.add(chunk_state)
                
                session.commit()
                
        except SQLAlchemyError as e:
            LOG.error(f"Database error updating file state for {file_path}: {e}")
            raise

    async def delete_file_state(self, file_path: Path) -> Set[str]:
        """Delete file state and return associated chunk IDs"""
        try:
            with Session(self.engine) as session:
                file_state = session.execute(
                    select(FileStateModel).where(
                        FileStateModel.file_path == str(file_path)
                    )
                ).scalar_one_or_none()
                
                if file_state:
                    chunk_ids = {chunk.chunk_id for chunk in file_state.chunks}
                    session.delete(file_state)
                    session.commit()
                    return chunk_ids
                
                return set()
                
        except SQLAlchemyError as e:
            LOG.error(f"Database error deleting file state for {file_path}: {e}")
            raise

    async def get_all_file_states(self) -> Dict[Path, FileStateBase]:
        """Get all file states"""
        try:
            with Session(self.engine) as session:
                stmt = select(FileStateModel)
                results = session.execute(stmt).scalars().all()
                
                return {
                    Path(state.file_path): FileStateBase(
                        file_path=state.file_path,
                        last_modified=state.last_modified,
                        file_hash=state.file_hash,
                        chunk_ids={chunk.chunk_id for chunk in state.chunks}
                    )
                    for state in results
                }
                
        except SQLAlchemyError as e:
            LOG.error(f"Database error getting all file states: {e}")
            raise

    async def detect_changes(self, current_files: Set[Path]) -> List[FileChange]:
        """Detect changes in files since last update"""
        try:
            changes = []
            existing_states = await self.get_all_file_states()
            
            # Check for new and modified files
            for file_path in current_files:
                if not file_path.exists():
                    continue
                    
                current_hash = self._compute_file_hash(file_path)
                current_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                if file_path not in existing_states:
                    changes.append(FileChange(
                        path=file_path,
                        change_type=StateChangeType.ADDED
                    ))
                else:
                    state = existing_states[file_path]
                    if (state.file_hash != current_hash or 
                        state.last_modified != current_mtime):
                        changes.append(FileChange(
                            path=file_path,
                            change_type=StateChangeType.MODIFIED,
                            old_chunks=state.chunk_ids
                        ))
            
            # Check for deleted files
            for file_path in existing_states:
                if file_path not in current_files:
                    changes.append(FileChange(
                        path=file_path,
                        change_type=StateChangeType.DELETED,
                        old_chunks=existing_states[file_path].chunk_ids
                    ))
            
            return changes
            
        except Exception as e:
            LOG.error(f"Error detecting changes: {e}")
            raise