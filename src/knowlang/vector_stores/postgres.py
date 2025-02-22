from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

import asyncpg

from knowlang.vector_stores.base import (SearchResult, VectorStore,
                                         VectorStoreError,
                                         VectorStoreInitError)

if TYPE_CHECKING:
    from knowlang.configs import DBConfig


class PostgresVectorStore(VectorStore):
    """Postgres implementation of VectorStore compatible with pgvector extension."""

    def __init__(
        self,
        connection_string: str,
        table_name: str,
        embedding_dim: int = 1536,
        similarity_metric: Literal['cosine'] = 'cosine'
    ):
        self.connection_string = connection_string
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.similarity_metric = similarity_metric
        self.pool: Optional[asyncpg.pool.Pool] = None

    def initialize(self) -> None:
        """Synchronously initialize the Postgres connection pool and ensure the vector store table exists."""
        try:
            asyncio.run(self._initialize())
        except Exception as e:
            raise VectorStoreInitError(f"Failed to initialize PostgresVectorStore: {str(e)}") from e

    async def _initialize(self) -> None:
        self.pool = await asyncpg.create_pool(dsn=self.connection_string)
        async with self.pool.acquire() as conn:
            # Ensure the pgvector extension is available.
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            # Create the table if it doesn't exist.
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id TEXT PRIMARY KEY,
                document TEXT,
                embedding vector({self.embedding_dim}),
                metadata JSONB
            );
            """
            await conn.execute(create_table_query)

    @classmethod
    def create_from_config(cls, config: DBConfig) -> "PostgresVectorStore":
        if not config.connection_url:
            raise VectorStoreInitError("Connection url not set for PostgresVectorStore.")
        return cls(
            connection_string=config.state_store.connection_url,
            table_name=config.collection_name,
            similarity_metric=config.similarity_metric
        )

    async def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        if self.pool is None:
            raise VectorStoreError("PostgresVectorStore is not initialized.")
        if ids is None:
            ids = [str(i) for i in range(len(documents))]
        async with self.pool.acquire() as conn:
            insert_query = f"""
            INSERT INTO {self.table_name} (id, document, embedding, metadata)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (id) DO NOTHING;
            """
            for doc, emb, meta, id_ in zip(documents, embeddings, metadatas, ids):
                await conn.execute(insert_query, id_, doc, emb, meta)

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        if self.pool is None:
            raise VectorStoreError("PostgresVectorStore is not initialized.")
        async with self.pool.acquire() as conn:
            # Use the pgvector distance operator (<=>) for similarity search.
            search_query = f"""
            SELECT id, document, metadata, (embedding <=> $1) AS distance
            FROM {self.table_name}
            ORDER BY embedding <=> $1
            LIMIT $2;
            """
            records = await conn.fetch(search_query, query_embedding, top_k)
            results = []
            for record in records:
                # Convert distance to a similarity score (this conversion may vary).
                score = 1.0 - record["distance"]
                if score_threshold is None or score >= score_threshold:
                    results.append(SearchResult(
                        document=record["document"],
                        metadata=record["metadata"],
                        score=score
                    ))
            return results

    async def delete(self, ids: List[str]) -> None:
        if self.pool is None:
            raise VectorStoreError("PostgresVectorStore is not initialized.")
        async with self.pool.acquire() as conn:
            delete_query = f"DELETE FROM {self.table_name} WHERE id = ANY($1::text[]);"
            await conn.execute(delete_query, ids)

    async def get_document(self, id: str) -> Optional[SearchResult]:
        if self.pool is None:
            raise VectorStoreError("PostgresVectorStore is not initialized.")
        async with self.pool.acquire() as conn:
            query = f"SELECT id, document, metadata FROM {self.table_name} WHERE id = $1;"
            record = await conn.fetchrow(query, id)
            if record:
                return SearchResult(
                    document=record["document"],
                    metadata=record["metadata"],
                    score=1.0  # Assuming direct retrieval is a perfect match.
                )
            return None

    async def update_document(
        self,
        id: str,
        document: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> None:
        if self.pool is None:
            raise VectorStoreError("PostgresVectorStore is not initialized.")
        async with self.pool.acquire() as conn:
            update_query = f"""
            UPDATE {self.table_name}
            SET document = $2, embedding = $3, metadata = $4
            WHERE id = $1;
            """
            await conn.execute(update_query, id, document, embedding, metadata)

    async def get_all(self) -> List[SearchResult]:
        if self.pool is None:
            raise VectorStoreError("PostgresVectorStore is not initialized.")
        async with self.pool.acquire() as conn:
            query = f"SELECT id, document, metadata FROM {self.table_name};"
            records = await conn.fetch(query)
            results = [
                SearchResult(
                    document=record["document"],
                    metadata=record["metadata"],
                    score=1.0
                )
                for record in records
            ]
            return results