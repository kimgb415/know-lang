from pydantic import BaseModel
from typing import Optional
from enum import Enum

class ChunkType(str, Enum):
    CLASS = "class"
    FUNCTION = "function"
    OTHER = "other"

class CodeChunk(BaseModel):
    """Represents a chunk of code with its metadata"""
    type: ChunkType
    content: str
    start_line: int
    end_line: int
    file_path: str
    name: Optional[str] = None
    parent_name: Optional[str] = None  # For nested classes/functions
    docstring: Optional[str] = None


class ModelProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"