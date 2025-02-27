from __future__ import annotations

from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

from chromadb.config import Settings

import chromadb
from knowlang.core.types import VectorStoreProvider
from knowlang.vector_stores.base import (SearchResult, VectorStore,
                                         VectorStoreInitError,
                                         register_vector_store)

if TYPE_CHECKING:
    from knowlang.configs import DBConfig, EmbeddingConfig

@register_vector_store(VectorStoreProvider.CHROMA)
class ChromaVectorStore(VectorStore):
    """ChromaDB implementation of VectorStore"""

    @classmethod
    def create_from_config(cls, config: DBConfig, embedding_config: EmbeddingConfig = None) -> "ChromaVectorStore":
        return cls(
            persist_directory=config.persist_directory,
            collection_name=config.collection_name,
            similarity_metric=config.similarity_metric
        )

    def accumulate_result(
        self,
        acc: List[SearchResult], 
        record: Tuple[str, float, Dict[str, Any]],
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        doc, meta, dist = record
        score = 1.0 - dist  # Convert distance to similarity score
        if score_threshold is None or score >= score_threshold:
            acc.append(SearchResult(
                document=doc,
                metadata=meta,
                score=score
            ))
        return acc

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
            self.collection = self.client.get_or_create_collection(
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
        self.assert_initialized()
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids or [str(i) for i in range(len(documents))]
        )

    async def query(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['metadatas', 'documents', 'distances']
        )
        return zip_longest(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0],
            fillvalue={}
        )

    async def delete(self, ids: List[str]) -> None:
        self.assert_initialized()
        self.collection.delete(ids=ids)

    async def get_document(self, id: str) -> Optional[SearchResult]:
        self.assert_initialized()
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
        self.assert_initialized()
        self.collection.upsert(
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
