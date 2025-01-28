from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path

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

class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        env_nested_delimiter='__'
    )
    
    llm: LLMConfig = Field(default_factory=LLMConfig)
    db: DBConfig = Field(default_factory=DBConfig)
    chunk_max_size: int = Field(
        default=1500,
        description="Maximum size of code chunks before splitting"
    )