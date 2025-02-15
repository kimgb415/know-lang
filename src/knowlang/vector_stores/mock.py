from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import numpy as np

from knowlang.vector_stores.base import VectorStore, SearchResult, VectorStoreError

@dataclass
class MockVectorStore(VectorStore):
    """Mock vector store for testing with controllable behavior"""
    
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
    
    async def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Mock adding documents"""
        if self.add_error:
            raise self.add_error
            
        doc_ids = ids or [str(i) for i in range(len(documents))]
        
        for doc_id, doc, emb, meta in zip(doc_ids, documents, embeddings, metadatas):
            self.documents[doc_id] = doc
            self.metadata[doc_id] = meta
            self.embeddings[doc_id] = emb

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """Mock vector search with customizable behavior"""
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
        """Mock document deletion"""
        if self.delete_error:
            raise self.delete_error
            
        for doc_id in ids:
            self.documents.pop(doc_id, None)
            self.metadata.pop(doc_id, None)
            self.embeddings.pop(doc_id, None)

    async def get_document(self, id: str) -> Optional[SearchResult]:
        """Mock document retrieval"""
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
        """Mock document update"""
        if self.update_error:
            raise self.update_error
            
        if id not in self.documents:
            raise VectorStoreError(f"Document {id} not found")
            
        self.documents[id] = document
        self.metadata[id] = metadata
        self.embeddings[id] = embedding

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