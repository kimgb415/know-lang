from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

import vecs
from vecs.collection import Record

from knowlang.vector_stores.base import (SearchResult, VectorStore,
                                         VectorStoreError,
                                         VectorStoreInitError)

if TYPE_CHECKING:
    from knowlang.configs import DBConfig, EmbeddingConfig


class PostgresVectorStore(VectorStore):
    """Postgres implementation of VectorStore compatible with the pgvector extension using psycopg."""

    @classmethod
    def create_from_config(cls, config: DBConfig, embedding_config: EmbeddingConfig) -> "PostgresVectorStore":
        if not config.connection_url:
            raise VectorStoreInitError("Connection url not set for PostgresVectorStore.")
        return cls(
            connection_string=config.connection_url,
            table_name=config.collection_name,
            embedding_dim=embedding_config.dimension,
            similarity_metric=config.similarity_metric,
        )

    def __init__(
        self,
        connection_string: str,
        table_name: str,
        embedding_dim: int,
        similarity_metric: Literal['cosine'] = 'cosine'
    ):
        self.connection_string = connection_string
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.similarity_metric = similarity_metric
        self.collection = None

    def initialize(self) -> None:
        """Initialize the Postgres vector store client and create a collection of vectors."""
        try:
            self.measure() # Validate similarity metric
            vx = vecs.create_client(self.connection_string)
            self.collection = vx.get_or_createcollection(name=self.table_name, dimension=self.embedding_dim)
        except Exception as e:
            raise VectorStoreInitError(f"Failed to initialize PostgresVectorStore: {str(e)}") from e

    def measure(self) -> vecs.IndexMeasure:
        if "cosine" in self.similarity_metric:
            return vecs.IndexMeasure.cosine_distance
        if "l1" in self.similarity_metric:
            return vecs.IndexMeasure.l1_distance
        if "l2" in self.similarity_metric:
            return vecs.IndexMeasure.l2_distance
        if "product" in self.similarity_metric:
            return vecs.IndexMeasure.max_inner_product
        raise VectorStoreError(f"Unsupported similarity metric: {self.similarity_metric}")

    async def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        self.assert_initialized()
        if len(documents) != len(embeddings):
            raise VectorStoreError("Number of documents and embeddings must match.")
        if ids is None:
            ids = [str(i) for i in range(len(documents))]
        if len(documents) != len(ids):
            raise VectorStoreError("Number of documents and ids must match.")
        vectors = [(id, emb, meta) for id, emb, meta in zip(ids, embeddings, metadatas)]
        self.collection.upsert(records=vectors)
        self.collection.create_index(measure=self.measure())

    def accumulate_result(
        self,
        acc: List[SearchResult], 
        record: Record, 
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        id, dist, meta = record
        score = 1.0 - dist  # Convert distance to similarity score
        if score_threshold is None or score >= score_threshold:
            acc.append(SearchResult(
                document=id,
                metadata=meta,
                score=score
            ))
        return acc

    async def query(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[SearchResult]:
        return self.collection.query(
            data=query_embedding,
            limit=top_k,
            measure=self.measure(),
            include_value=True,
            include_metadata=True
        )

    async def delete(self, ids: List[str]) -> None:
        self.assert_initialized()
        self.collection.delete(ids)

    async def get_document(self, id: str) -> Optional[SearchResult]:
        self.assert_initialized()
        results = self.collection.fetch(ids=[id])
        return results[0] if results else None

    async def update_document(
        self,
        id: str,
        document: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> None:
        self.assert_initialized()
        self.collection.upsert([(id, embedding, metadata)])

    async def get_all(self) -> List[SearchResult]:
        raise NotImplementedError("Postgres fetching all documents not implemented yet")