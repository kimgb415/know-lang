# src/knowlang/core/types.py
from enum import Enum
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, Field

from knowlang.vector_stores.base import VectorStore, VectorStoreNotFoundError
from knowlang.vector_stores.chroma import ChromaVectorStore
from knowlang.vector_stores.mock import MockVectorStore
from knowlang.vector_stores.postgres import PostgresVectorStore


class LanguageEnum(str, Enum):
    """Supported programming languages"""
    PYTHON = "python"
    CPP = "cpp"

class BaseChunkType(str, Enum):
    """Base chunk types common across languages"""
    CLASS = "class"
    FUNCTION = "function"
    OTHER = "other"

class CodeVisibility(str, Enum):
    """Access modifiers/visibility"""
    PUBLIC = "public"
    PRIVATE = "private"
    PROTECTED = "protected"
    DEFAULT = "default"

class CodeMetadata(BaseModel):
    """Base metadata that can be extended per language"""
    visibility: Optional[CodeVisibility] = CodeVisibility.DEFAULT
    is_static: bool = False
    is_abstract: bool = False
    is_template: bool = False
    namespace: Optional[str] = None
    # For language-specific metadata that doesn't fit the common fields
    language_specific: Dict[str, Any] = Field(default_factory=dict)

class CodeLocation(BaseModel):
    """Location information for a code chunk"""
    file_path: str
    start_line: int
    end_line: int

    def to_single_line(self) -> str:
        """Convert location to a single line string"""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

class CodeChunk(BaseModel):
    """Generic code chunk that works across languages"""
    type: BaseChunkType
    language: LanguageEnum
    location: CodeLocation
    content: str
    name: str
    docstring: Optional[str] = None
    metadata: CodeMetadata = Field(default_factory=CodeMetadata)
    
    def add_language_metadata(self, key: str, value: Any) -> None:
        """Add language-specific metadata"""
        self.metadata.language_specific[key] = value

class DatabaseChunkMetadata(BaseModel):
    """Metadata for database storage"""
    name: str
    type: str
    language: str
    start_line: int
    end_line: int
    file_path: str

    @classmethod
    def from_code_chunk(cls, chunk: CodeChunk) -> "DatabaseChunkMetadata":
        """Create a DatabaseChunkMetadata instance from a CodeChunk"""
        return cls(
            name=chunk.name,
            type=chunk.type,
            language=chunk.language,
            start_line=chunk.location.start_line,
            end_line=chunk.location.end_line,
            file_path=chunk.location.file_path
        )

class ModelProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    VOYAGE = "voyage"
    TESTING = "testing"

class VectorStoreProvider(Enum):
    CHROMA = ("chroma", ChromaVectorStore)
    POSTGRES = ("postgres", PostgresVectorStore)
    TESTING = ("testing", MockVectorStore)

    def __init__(self, value: str, store_class: Type[VectorStore]):
        self._value_ = value
        self._store_class = store_class

    @property
    def store_class(self) -> Type[VectorStore]:
        if self._store_class is None:
            raise VectorStoreNotFoundError(f"Provider {self._value_} not supported")
        return self._store_class

class StateStoreProvider(str, Enum):
    """Supported state store types"""
    SQLITE = "sqlite"
    POSTGRES = "postgres"
    CUSTOM = "custom"
    # add more types in future:
    # MYSQL = "mysql"
    # MONGODB = "mongodb"