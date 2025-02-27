from __future__ import annotations

from abc import ABC, abstractmethod
from functools import reduce
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

from pydantic import BaseModel

if TYPE_CHECKING:
    from knowlang.configs import DBConfig, EmbeddingConfig


# ----------------- Exceptions -----------------
class VectorStoreError(Exception):
    """Base exception for vector store errors"""
    pass

class VectorStoreInitError(VectorStoreError):
    """Error during vector store initialization"""
    pass

class VectorStoreNotFoundError(VectorStoreError):
    """Error when requested vector store provider is not found"""
    pass

# ----------------- Data Models -----------------
class SearchResult(BaseModel):
    """Standardized search result across vector stores"""
    document: str
    metadata: Dict[str, Any]
    score: float  # Similarity/relevance score

# ----------------- Abstract Base Class -----------------
class VectorStore(ABC):
    """Abstract base class for vector store implementations"""

    def __init__(self, *args, **kwargs):
        self.collection = kwargs.get('collection', None)

    def assert_initialized(self) -> None:
        """Assert that the vector store is initialized"""
        if self.collection is None:
            raise VectorStoreError(f"{self.__class__.__name__} is not initialized.")

    @classmethod
    @abstractmethod
    def create_from_config(cls, db_config: DBConfig, embedding_config: EmbeddingConfig) -> VectorStore:
        """
        Create a VectorStore instance from configuration.
        """
        pass

    @classmethod
    @abstractmethod
    def initialize(cls) -> None:
        """
        Initialize the vector store (this might create indices, load data, etc.).
        """
        pass

    @abstractmethod
    def accumulate_result(
        self,
        acc: List[SearchResult],
        record: Any, 
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """Accumulate search result"""
        pass

    @abstractmethod
    async def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Add documents with their embeddings and metadata"""
        pass

    @abstractmethod
    async def query(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Any]:
        """Query the vector store for similar documents"""
        pass

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """Search for similar documents"""
        self.assert_initialized()
        records = await self.query(query_embedding=query_embedding, top_k=top_k)
        return reduce(
            lambda acc, record: self.accumulate_result(acc, record, score_threshold),
            records,
            []
        )

    @abstractmethod
    async def delete(self, ids: List[str]) -> None:
        """Delete documents by their IDs"""
        pass

    @abstractmethod
    async def get_document(self, id: str) -> Optional[SearchResult]:
        """Retrieve a single document by ID"""
        pass

    @abstractmethod
    async def update_document(
        self,
        id: str,
        document: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> None:
        """Update an existing document"""
        pass

    @abstractmethod
    async def get_all(self) -> List[SearchResult]:
        """Get all documents in the store"""
        pass

# Dictionary mapping provider keys to concrete VectorStore implementations.
VECTOR_STORE_REGISTRY: Dict[str, Type[VectorStore]] = {}

def register_vector_store(provider: str):
    """Decorator to register a vector store implementation for a given provider key."""
    def decorator(cls: Type[VectorStore]):
        VECTOR_STORE_REGISTRY[provider] = cls
        return cls
    return decorator

def get_vector_store(config: DBConfig, embedding_config: EmbeddingConfig) -> VectorStore:
    """
    Factory method that retrieves a vector store instance based on configuration.
    
    Args:
        config: Database configuration
        
    Returns:
        Initialized vector store instance
        
    Raises:
        VectorStoreInitError: If initialization fails
    """
    try:
        provider = config.db_provider
        if provider not in VECTOR_STORE_REGISTRY:
            raise VectorStoreNotFoundError(f"Vector store provider '{provider}' is not registered.")
        store_cls = VECTOR_STORE_REGISTRY[provider]
        vector_store = store_cls.create_from_config(config, embedding_config)
        vector_store.initialize()
        return vector_store
    except VectorStoreError:
        # Re-raise VectorStoreError subclasses as-is
        raise
    except Exception as e:
        # Wrap any other exceptions
        raise VectorStoreInitError(f"Failed to create vector store: {str(e)}") from e