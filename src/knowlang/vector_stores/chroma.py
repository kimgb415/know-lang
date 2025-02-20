
from pathlib import Path
import chromadb
from typing import List, Dict, Any, Literal, Optional
from chromadb.config import Settings
from chromadb.errors import InvalidCollectionException

from knowlang.vector_stores.base import VectorStore, SearchResult, VectorStoreError, VectorStoreInitError

class ChromaVectorStore(VectorStore):
    """ChromaDB implementation of VectorStore"""
    
    def __init__(
        self, 
        persist_directory: Path,
        collection_name: str,
        similarity_metric: Literal['cosine'] = 'cosine'
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.similarity_metric = similarity_metric
        self.client = None
        self.collection = None
        
    def initialize(self) -> None:
        """Initialize ChromaDB client and collection"""
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                )
            except InvalidCollectionException:  # Collection doesn't exist
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": self.similarity_metric}
                )
                
        except Exception as e:
            raise VectorStoreInitError(f"Failed to initialize ChromaDB: {str(e)}") from e

    async def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        if not self.collection:
            raise VectorStoreError("ChromaDB collection not initialized")
            
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids or [str(i) for i in range(len(documents))]
        )

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        if not self.collection:
            raise VectorStoreError("ChromaDB collection not initialized")
            
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['metadatas', 'documents', 'distances']
        )
        
        search_results = []
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            if score_threshold is None or dist <= score_threshold:
                search_results.append(SearchResult(
                    document=doc,
                    metadata=meta,
                    score=1.0 - dist  # Convert distance to similarity score
                ))
        
        return search_results

    async def delete(self, ids: List[str]) -> None:
        if not self.collection:
            raise VectorStoreError("ChromaDB collection not initialized")
        self.collection.delete(ids=ids)

    async def get_document(self, id: str) -> Optional[SearchResult]:
        if not self.collection:
            raise VectorStoreError("ChromaDB collection not initialized")
            
        try:
            result = self.collection.get(ids=[id])
            if result['documents']:
                return SearchResult(
                    document=result['documents'][0],
                    metadata=result['metadatas'][0],
                    score=1.0  # Perfect match for direct retrieval
                )
        except ValueError:
            return None

    async def update_document(
        self,
        id: str,
        document: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> None:
        if not self.collection:
            raise VectorStoreError("ChromaDB collection not initialized")
            
        self.collection.update(
            ids=[id],
            documents=[document],
            embeddings=[embedding],
            metadatas=[metadata]
        )
    
    async def get_all(self) -> List[SearchResult]:
        raise NotImplementedError("ChromaDB fetching all documents not implemented yet")

'''
Action Items:
- Track local changes through git commits (assuming you have a git repository)
```bash
git diff --name-only HEAD~10
```
We can run cron job every 10 minutes to check last 10 commits.
We can further run other git commands to check that those commits occurred in the last 10 minutes.
- Update vector store if changes are detected
'''
