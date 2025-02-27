from __future__ import annotations

from typing import TYPE_CHECKING

from knowlang.vector_stores.base import VectorStore, get_vector_store

if TYPE_CHECKING:
    from knowlang.configs import DBConfig, EmbeddingConfig


class VectorStoreFactory:
    """Factory for creating vector store instances"""
    
    @staticmethod
    def get(
        config: DBConfig,
        embedding_config: EmbeddingConfig
    ) -> VectorStore:
        get_vector_store(config, embedding_config)