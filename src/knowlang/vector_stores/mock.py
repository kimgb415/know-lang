from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import AsyncMock

import numpy as np

from knowlang.vector_stores import SearchResult, VectorStore, VectorStoreError


@dataclass
class MockVectorStore(VectorStore):
    """Mock vector store for testing with controllable behavior and tracking of method calls"""
    
    # Store documents and their metadata
    documents: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    embeddings: Dict[str, List[float]] = field(default_factory=dict)
    
    # Optional error injection for testing error scenarios
    search_error: Optional[Exception] = None
    add_error: Optional[Exception] = None
    delete_error: Optional[Exception] = None
    update_error: Optional[Exception] = None
    
    # Optional mock behavior functions
    mock_search_fn: Optional[Callable] = None
    
    # Tracking for test verification
    deleted_chunks: List[str] = field(default_factory=list)
    added_documents: List[str] = field(default_factory=list)
    updated_documents: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Set up tracking mocks for call verification"""
        # Create wrappers for method call tracking
        # For test assertions if needed
        self.add_documents_mock = AsyncMock(side_effect=self._add_documents)
        self.search_mock = AsyncMock(side_effect=self._search)
        self.delete_mock = AsyncMock(side_effect=self._delete)
        self.get_document_mock = AsyncMock(side_effect=self._get_document)
        self.update_document_mock = AsyncMock(side_effect=self._update_document)
        self.get_all_mock = AsyncMock(side_effect=self._get_all)
    
    async def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Mock adding documents with call tracking"""
        return await self.add_documents_mock(documents, embeddings, metadatas, ids)
    
    async def _add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Actual implementation of add_documents"""
        if self.add_error:
            raise self.add_error
            
        doc_ids = ids or [str(i) for i in range(len(documents))]
        
        for doc_id, doc, emb, meta in zip(doc_ids, documents, embeddings, metadatas):
            self.documents[doc_id] = doc
            self.metadata[doc_id] = meta
            self.embeddings[doc_id] = emb
            self.added_documents.append(doc_id)

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """Mock vector search with call tracking"""
        return await self.search_mock(query_embedding, top_k, score_threshold)
    
    async def _search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """Actual implementation of search"""
        if self.search_error:
            raise self.search_error
            
        if self.mock_search_fn:
            return await self.mock_search_fn(query_embedding, top_k, score_threshold)
            
        # Default behavior: return documents sorted by cosine similarity
        scores = {}
        query_vec = np.array(query_embedding)
        
        for doc_id, doc_vec in self.embeddings.items():
            # Compute cosine similarity
            similarity = np.dot(query_vec, doc_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
            )
            scores[doc_id] = similarity
            
        # Sort by similarity and apply threshold
        sorted_ids = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        results = []
        
        for doc_id in sorted_ids[:top_k]:
            if score_threshold is None or scores[doc_id] >= score_threshold:
                results.append(SearchResult(
                    document=self.documents[doc_id],
                    metadata=self.metadata[doc_id],
                    score=float(scores[doc_id])
                ))
                
        return results

    async def delete(self, ids: List[str]) -> None:
        """Mock document deletion with call tracking"""
        return await self.delete_mock(ids)
    
    async def _delete(self, ids: List[str]) -> None:
        """Actual implementation of delete"""
        if self.delete_error:
            raise self.delete_error
        
        # Track deleted IDs for verification
        self.deleted_chunks.extend(ids)
            
        for doc_id in ids:
            self.documents.pop(doc_id, None)
            self.metadata.pop(doc_id, None)
            self.embeddings.pop(doc_id, None)

    async def get_document(self, id: str) -> Optional[SearchResult]:
        """Mock document retrieval with call tracking"""
        return await self.get_document_mock(id)
    
    async def _get_document(self, id: str) -> Optional[SearchResult]:
        """Actual implementation of get_document"""
        if id not in self.documents:
            return None
            
        return SearchResult(
            document=self.documents[id],
            metadata=self.metadata[id],
            score=1.0
        )

    async def update_document(
        self,
        id: str,
        document: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> None:
        """Mock document update with call tracking"""
        return await self.update_document_mock(id, document, embedding, metadata)
    
    async def _update_document(
        self,
        id: str,
        document: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> None:
        """Actual implementation of update_document"""
        if self.update_error:
            raise self.update_error
            
        if id not in self.documents:
            raise VectorStoreError(f"Document {id} not found")
        
        # Track updated documents
        self.updated_documents.append(id)
            
        self.documents[id] = document
        self.metadata[id] = metadata
        self.embeddings[id] = embedding
    
    async def get_all(self) -> List[SearchResult]:
        """Return all documents with call tracking"""
        return await self.get_all_mock()
    
    async def _get_all(self) -> List[SearchResult]:
        """Actual implementation of get_all"""
        return [
            SearchResult(document=self.documents[doc_id], metadata=self.metadata[doc_id], score=1.0)
            for doc_id in self.metadata.keys()
        ]

    def reset(self):
        """Reset the mock store to empty state"""
        self.documents.clear()
        self.metadata.clear()
        self.embeddings.clear()
        self.search_error = None
        self.add_error = None
        self.delete_error = None
        self.update_error = None
        self.mock_search_fn = None
        self.deleted_chunks.clear()
        self.added_documents.clear()
        self.updated_documents.clear()
        
        # Reset call counts
        self.add_documents_mock.reset_mock()
        self.search_mock.reset_mock()
        self.delete_mock.reset_mock()
        self.get_document_mock.reset_mock()
        self.update_document_mock.reset_mock()
        self.get_all_mock.reset_mock()