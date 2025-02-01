import ollama
import openai
from know_lang_bot.config import EmbeddingConfig, ModelProvider
from typing import Union, List, overload

# Type definitions
EmbeddingVector = List[float]

class EmbeddingConfig:
    def __init__(self, provider: ModelProvider, model_name: str):
        self.provider = provider
        self.model_name = model_name

def _process_ollama_batch(inputs: List[str], model_name: str) -> List[EmbeddingVector]:
    """Helper function to process Ollama embeddings in batch."""
    return [
        ollama.embed(model=model_name, input=inputs)['embeddings']
    ]

def _process_openai_batch(inputs: List[str], model_name: str) -> List[EmbeddingVector]:
    """Helper function to process OpenAI embeddings in batch."""
    response = openai.embeddings.create(
        input=inputs,
        model=model_name
    )
    return [item.embedding for item in response.data]

@overload
def generate_embedding(input: str, config: EmbeddingConfig) -> EmbeddingVector: ...

@overload
def generate_embedding(input: List[str], config: EmbeddingConfig) -> List[EmbeddingVector]: ...

def generate_embedding(
    input: Union[str, List[str]], 
    config: EmbeddingConfig
) -> Union[EmbeddingVector, List[EmbeddingVector]]:
    """
    Generate embeddings for single text input or batch of texts.
    
    Args:
        input: Single string or list of strings to embed
        config: Configuration object containing provider and model information
    
    Returns:
        Single embedding vector for single input, or list of embedding vectors for batch input
    
    Raises:
        ValueError: If input type is invalid or provider is not supported
        RuntimeError: If embedding generation fails
    """
    if not input:
        raise ValueError("Input cannot be empty")

    # Convert single string to list for batch processing
    is_single_input = isinstance(input, str)
    inputs = [input] if is_single_input else input

    try:
        if config.provider == ModelProvider.OLLAMA:
            embeddings = _process_ollama_batch(inputs, config.model_name)
        elif config.provider == ModelProvider.OPENAI:
            embeddings = _process_openai_batch(inputs, config.model_name)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")

        # Return single embedding for single input
        return embeddings[0] if is_single_input else embeddings

    except Exception as e:
        raise RuntimeError(f"Failed to generate embeddings: {str(e)}") from e