import fnmatch
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings

from knowlang.core.types import ModelProvider, VectorStoreProvider

from .base import generate_model_config
from .chat_config import ChatbotAnalyticsConfig, ChatConfig
from .state_store_config import StateStoreConfig


def _validate_api_key(v: Optional[str], info: ValidationInfo) -> Optional[str]:
    """Validate API key is present when required"""
    if info.data['model_provider'] in [
        ModelProvider.OPENAI, 
        ModelProvider.ANTHROPIC,
        ModelProvider.VOYAGE
    ]:
        if not v:
            raise ValueError(f"API key required for {info.data['model_provider']}")
        elif info.data['model_provider'] == ModelProvider.ANTHROPIC:
            os.environ["ANTHROPIC_API_KEY"] = v
        elif info.data['model_provider'] == ModelProvider.OPENAI:
            os.environ["OPENAI_API_KEY"] = v
        elif info.data['model_provider'] == ModelProvider.VOYAGE:
            os.environ["VOYAGE_API_KEY"] = v
            
    return v

class PathPatterns(BaseSettings):
    include: List[str] = Field(
        default=["**/*"],
        description="Glob patterns for paths to include"
    )
    exclude: List[str] = Field(
        default=[
            "**/venv/**", 
            "**/.git/**", 
            "**/__pycache__/**", 
            "**/tests/**",
        ],
        description="Glob patterns for paths to exclude"
    )

    def should_process_path(self, path: str) -> bool:
        """Check if a path should be processed based on include/exclude patterns"""
        path_str = str(path)
        
        # First check exclusions
        for pattern in self.exclude:
            if fnmatch.fnmatch(path_str, pattern):
                return False
        
        # Then check inclusions
        for pattern in self.include:
            if fnmatch.fnmatch(path_str, pattern):
                return True
        
        return False

class LanguageConfig(BaseSettings):
    enabled: bool = True
    file_extensions: List[str]
    tree_sitter_language: str
    chunk_types: List[str]
    max_file_size: int = Field(
        default=1_000_000,  # 1MB
        description="Maximum file size to process in bytes"
    )

class ParserConfig(BaseSettings):
    languages: Dict[str, LanguageConfig] = Field(
        default={
            "python": LanguageConfig(
                file_extensions=[".py"],
                tree_sitter_language="python",
                chunk_types=["class_definition", "function_definition"]
            )
        }
    )
    path_patterns: PathPatterns = Field(default_factory=PathPatterns)
    enable_code_summarization: bool = Field(
        default=False,
        description="Enable code summarization to be stored in the vector store"
    )


class EmbeddingConfig(BaseSettings):
    """Shared embedding configuration"""
    model_name: str = Field(
        default="mxbai-embed-large",
        description="Name of the embedding model"
    )
    model_provider: ModelProvider = Field(
        default=ModelProvider.OLLAMA,
        description="Provider for embeddings"
    )
    dimension: int = Field(
        default=768,
        description="Embedding dimension"
    )
    settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific settings"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for the model provider"
    )

    @field_validator('api_key', mode='after')
    @classmethod
    def validate_api_key(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        return _validate_api_key(v, info)

class LLMConfig(BaseSettings):
    model_name: str = Field(
        default="llama3.2",
        description="Name of the LLM model to use"
    )
    model_provider: str = Field(
        default=ModelProvider.OLLAMA,
        description="Model provider (anthropic, openai, ollama, etc)"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for the model provider"
    )
    model_settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional model settings"
    )

    @field_validator('api_key', mode='after')
    @classmethod
    def validate_api_key(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        return _validate_api_key(v, info)

class DBConfig(BaseSettings):
    db_provider: VectorStoreProvider = Field(
        default=VectorStoreProvider.CHROMA,
        description="Vector Database provider"
    )
    connection_url: Optional[str] = Field(
        default=None,
        description="Database connection URL (for network-based stores like PostgreSQL)"
    )
    persist_directory: Path = Field(
        default=Path("./vectordb"),
        description="Directory to vector store"
    )
    collection_name: str = Field(
        default="code",
        description="Name of the vector store collection"
    )
    codebase_directory: Path = Field(
        default=Path("./"),
        description="Root directory of the codebase to analyze"
    )
    codebase_url: Optional[str] = Field(
        default=None,
        description="URL of the codebase repository"
    )
    similarity_metric: Literal['cosine'] = Field(
        default='cosine',
        description="Similarity metric for vector search"
    )
    state_store: StateStoreConfig = Field(default_factory=StateStoreConfig)

class RerankerConfig(BaseSettings):
    enabled: bool = Field(
        default=False,
        description="Enable reranking"
    )
    model_name: str = Field(
        default="reranker",
        description="Name of the reranker model to use"
    )
    model_provider: str = Field(
        default=ModelProvider.OLLAMA,
        description="Model provider (anthropic, openai, ollama, etc)"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for the model provider"
    )
    top_k: int = Field(
        default=4,
        description="Number of most relevant documents to return from reranking"
    )
    relevance_threshold: float = Field(
        default=0.5,
        description="Minimum relevance score to include a document in reranking"
    )

    @field_validator('api_key', mode='after')
    @classmethod
    def validate_api_key(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        return _validate_api_key(v, info)


class EvaluatorConfig(LLMConfig):
    evaluation_rounds: int = Field(
        default=1,
        description="Number of evaluation rounds per test case"
    )

class AppConfig(BaseSettings):
    model_config = generate_model_config(
        env_file=".env.app",
    )   
    llm: LLMConfig = Field(default_factory=LLMConfig)
    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    db: DBConfig = Field(default_factory=DBConfig)
    parser: ParserConfig = Field(default_factory=ParserConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chat_analytics: ChatbotAnalyticsConfig = Field(default_factory=ChatbotAnalyticsConfig)