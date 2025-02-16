from enum import Enum
from pathlib import Path
from typing import Literal

from knowlang.core.types import VectorStoreProvider
from knowlang.vector_stores.base import (
    VectorStore, 
    VectorStoreError, 
    VectorStoreInitError, 
    VectorStoreNotFoundError
)
from knowlang.vector_stores.chroma import ChromaVectorStore 
from knowlang.configs.config import DBConfig

class VectorStoreFactory:
    """Factory for creating vector store instances"""
    
    @staticmethod
    def get(
        config: DBConfig
    ) -> VectorStore:
        """
        Create and initialize a vector store instance
        
        Args:
            provider: Vector store provider to use
            config: Database configuration
            
        Returns:
            Initialized vector store instance
            
        Raises:
            VectorStoreNotFoundError: If provider is not supported
            VectorStoreInitError: If initialization fails
        """
        try:
            vector_store: VectorStore
            
            if config.db_provider == VectorStoreProvider.CHROMA:
                vector_store = ChromaVectorStore(
                    persist_directory=config.persist_directory,
                    collection_name=config.collection_name,
                    similarity_metric=config.similarity_metric
                )
            else:
                raise VectorStoreNotFoundError(f"Provider {config.db_provider} not supported")
            
            # Initialize the store
            vector_store.initialize()
            
            return vector_store
            
        except VectorStoreError:
            # Re-raise VectorStoreError subclasses as-is
            raise
        except Exception as e:
            # Wrap any other exceptions
            raise VectorStoreInitError(f"Failed to create vector store: {str(e)}") from e