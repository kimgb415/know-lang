import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from sqlalchemy import (Column, DateTime, ForeignKey, Integer, String,
                        create_engine, select)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from knowlang.configs import DBConfig
from knowlang.core.types import StateStoreProvider
from knowlang.indexing.file_utils import (compute_file_hash, get_absolute_path,
                                          get_relative_path)

from .base import (FileChange, FileState, StateChangeType, StateStore,
                   register_state_store)

LOG = logging.getLogger(__name__)
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

class ChunkStateModel(Base):
    """SQLAlchemy model for chunk states"""
    __tablename__ = 'chunk_states'
    
    id = Column(Integer, primary_key=True)
    chunk_id = Column(String, unique=True, index=True)
    file_id = Column(Integer, ForeignKey('file_states.id'))
    file = relationship("FileStateModel", back_populates="chunks")

@register_state_store(StateStoreProvider.SQLITE)
@register_state_store(StateStoreProvider.POSTGRES)
class SQLAlchemyStateStore(StateStore):
    """SQLAlchemy-based state storage implementation supporting both SQLite and PostgreSQL"""
    def __init__(self, config: DBConfig):
        """Initialize database with configuration and create schema if needed"""
        self.config = DBConfig.model_validate(config)
        
        # Validate store type
        if config.state_store.provider not in (StateStoreProvider.SQLITE, StateStoreProvider.POSTGRES):
            raise ValueError(f"Invalid store type: {config.state_store.provider}")
            
        # Initialize database connection
        connection_args = config.state_store.get_connection_args()
        self.engine = create_engine(
            connection_args.pop('url'),
            **connection_args
        )
        self.Session = sessionmaker(bind=self.engine)

        # Create database schema if it doesn't exist
        Base.metadata.create_all(self.engine)
        
        LOG.info(f"Initialized {config.state_store.provider} state store schema at {self.config.state_store.store_path}")

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

    async def get_file_state(self, file_path: Path) -> Optional[FileState]:
        """Get current state of a file"""
        try:
            with self.Session() as session:
                # Convert to relative path for storage/lookup
                relative_path = str(get_relative_path(file_path, self.config))
                
                stmt = select(FileStateModel).where(
                    FileStateModel.file_path == relative_path
                )
                result = session.execute(stmt).scalar_one_or_none()
                
                return (FileState(
                    file_path=result.file_path,
                    last_modified=result.last_modified,
                    file_hash=result.file_hash,
                    chunk_ids={chunk.chunk_id for chunk in result.chunks}
                ) if result else None)
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
            with self.Session() as session:
                # Ensure we're using relative path for storage
                relative_path = str(get_relative_path(file_path, self.config))
                
                # Compute new file hash
                file_hash = compute_file_hash(file_path)
                current_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                # Get or create file state
                file_state = session.execute(
                    select(FileStateModel).where(
                        FileStateModel.file_path == relative_path
                    )
                ).scalar_one_or_none()
                
                if not file_state:
                    file_state = FileStateModel(
                        file_path=relative_path,
                        last_modified=current_mtime,
                        file_hash=file_hash
                    )
                    session.add(file_state)
                else:
                    file_state.last_modified = current_mtime
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
            with self.Session() as session:
                # Ensure we're using relative path for storage
                relative_path = str(get_relative_path(file_path, self.config))
                
                file_state = session.execute(
                    select(FileStateModel).where(
                        FileStateModel.file_path == relative_path
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

    async def get_all_file_states(self) -> Dict[Path, FileState]:
        """Get all file states"""
        try:
            with self.Session() as session:
                stmt = select(FileStateModel)
                results = session.execute(stmt).scalars().all()
                
                return {
                    # Convert stored relative paths to absolute paths for comparisons
                    get_absolute_path(Path(state.file_path), self.config): FileState(
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
                    
                current_hash = compute_file_hash(file_path)
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