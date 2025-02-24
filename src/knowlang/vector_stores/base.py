from __future__ import annotations

from abc import ABC, abstractmethod
from functools import reduce
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel

if TYPE_CHECKING:
    from knowlang.configs import DBConfig


class VectorStoreError(Exception):
    """Base exception for vector store errors"""
    pass

class VectorStoreInitError(VectorStoreError):
    """Error during vector store initialization"""
    pass

class VectorStoreNotFoundError(VectorStoreError):
    """Error when requested vector store provider is not found"""
    pass

class SearchResult(BaseModel):
    """Standardized search result across vector stores"""
    document: str
    metadata: Dict[str, Any]
    score: float  # Similarity/relevance score

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
    def create_from_config(config: DBConfig) -> "VectorStore":
        """Create a VectorStore instance from configuration"""
        pass

    @classmethod
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the vector store"""
        pass

    @abstractmethod
    def accumulate_result(
        self,
        acc: List[SearchResult],
        record: Any, 
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """Accumulate search results"""
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
        records = await self.query(
            query_embedding=query_embedding,
            top_k=top_k,
            score_threshold=score_threshold
        )
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

