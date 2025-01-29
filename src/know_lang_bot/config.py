from typing import Optional, Dict, Any, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path
import fnmatch

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

    def supports_extension(self, ext: str) -> bool:
        """Check if the language supports a given file extension"""
        return ext in self.file_extensions

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


class LLMConfig(BaseSettings):
    model_name: str = Field(
        default="llama3.2",
        description="Name of the LLM model to use"
    )
    model_provider: str = Field(
        default="ollama",
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
    embedding_model: str = Field(
        default="mxbai-embed-large",
        description="Name of the embedding model to use"
    )
    embedding_provider: str = Field(
        default="ollama",
        description="Provider for embeddings (ollama, openai, etc)"
    )

class DBConfig(BaseSettings):
    persist_directory: Path = Field(
        default=Path("./chroma_db"),
        description="Directory to store ChromaDB files"
    )
    collection_name: str = Field(
        default="code_chunks",
        description="Name of the ChromaDB collection"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        description="Embedding model to use"
    )
    codebase_directory: Path = Field(
        default=Path("./"),
        description="Root directory of the codebase to analyze"
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