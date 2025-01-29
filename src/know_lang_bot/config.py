from typing import Optional, Dict, Any, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, ValidationInfo
from pathlib import Path
import fnmatch
from know_lang_bot.core.types import ModelProvider

class PathPatterns(BaseSettings):
    include: List[str] = Field(
        default=["**/*"],
        description="Glob patterns for paths to include"
    )
    exclude: List[str] = Field(
        default=["**/venv/**", "**/.git/**", "**/__pycache__/**"],
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


class EmbeddingConfig(BaseSettings):
    """Shared embedding configuration"""
    model_name: str = Field(
        default="mxbai-embed-large",
        description="Name of the embedding model"
    )
    provider: ModelProvider = Field(
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
        """Validate API key is present when required"""
        if info.data['model_provider'] in [ModelProvider.OPENAI, ModelProvider.ANTHROPIC] and not v:
            raise ValueError(f"API key required for {info.data['model_provider']}")
        return v

class DBConfig(BaseSettings):
    persist_directory: Path = Field(
        default=Path("./chroma_db"),
        description="Directory to store ChromaDB files"
    )
    collection_name: str = Field(
        default="code_chunks",
        description="Name of the ChromaDB collection"
    )
    codebase_directory: Path = Field(
        default=Path("./"),
        description="Root directory of the codebase to analyze"
    )

class ChatConfig(BaseSettings):
    max_context_chunks: int = Field(
        default=5,
        description="Maximum number of similar chunks to include in context"
    )
    similarity_threshold: float = Field(
        default=0.7,
        description="Minimum similarity score to include a chunk"
    )
    interface_title: str = Field(
        default="Code Repository Q&A Assistant",
        description="Title shown in the chat interface"
    )
    interface_description: str = Field(
        default="Ask questions about the codebase and I'll help you understand it!",
        description="Description shown in the chat interface"
    )

class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        env_nested_delimiter='__'
    )
    
    llm: LLMConfig = Field(default_factory=LLMConfig)
    db: DBConfig = Field(default_factory=DBConfig)
    parser: ParserConfig = Field(default_factory=ParserConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)