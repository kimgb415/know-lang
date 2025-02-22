from __future__ import annotations

from abc import ABC, abstractmethod
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

    @classmethod
    @abstractmethod
    def create_from_config(config: DBConfig) -> "VectorStore":
        """Create a VectorStore instance from configuration"""
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
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """Search for similar documents"""
        pass
    
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

