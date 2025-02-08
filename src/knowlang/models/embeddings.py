import ollama
import openai
import voyageai
import voyageai.client
from knowlang.configs.config import EmbeddingConfig, ModelProvider
from typing import Union, List, overload, Optional
from enum import Enum

# Type definitions
EmbeddingVector = List[float]


class EmbeddingInputType(Enum):
    DOCUMENT = "document"
    QUERY = "query"


def _process_ollama_batch(inputs: List[str], model_name: str) -> List[EmbeddingVector]:
    """Helper function to process Ollama embeddings in batch."""
    return ollama.embed(model=model_name, input=inputs)['embeddings']
    

def _process_openai_batch(inputs: List[str], model_name: str) -> List[EmbeddingVector]:
    """Helper function to process OpenAI embeddings in batch."""
    response = openai.embeddings.create(
        input=inputs,
        model=model_name
    )
    return [item.embedding for item in response.data]

def _process_voiage_batch(inputs: List[str], model_name: str, input_type:EmbeddingInputType) -> List[EmbeddingVector]:
    """Helper function to process VoyageAI embeddings in batch."""
    vo = voyageai.Client()
    embeddings_obj = vo.embed(model=model_name, texts=inputs, input_type=input_type.value)
    return embeddings_obj.embeddings

@overload
def generate_embedding(input: str, config: EmbeddingConfig, input_type: Optional[EmbeddingInputType]) -> EmbeddingVector: ...

@overload
def generate_embedding(input: List[str], config: EmbeddingConfig, input_type: Optional[EmbeddingInputType]) -> List[EmbeddingVector]: ...

def generate_embedding(
    input: Union[str, List[str]], 
    config: EmbeddingConfig,
    input_type: Optional[EmbeddingInputType] = EmbeddingInputType.DOCUMENT
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
        if config.model_provider == ModelProvider.OLLAMA:
            embeddings = _process_ollama_batch(inputs, config.model_name)
        elif config.model_provider == ModelProvider.OPENAI:
            embeddings = _process_openai_batch(inputs, config.model_name)
        elif config.model_provider == ModelProvider.VOYAGE:
            embeddings = _process_voiage_batch(inputs, config.model_name, input_type)
        else:
            raise ValueError(f"Unsupported provider: {config.model_provider}")

        # Return single embedding for single input
        return embeddings[0] if is_single_input else embeddings

    except Exception as e:
        raise RuntimeError(f"Failed to generate embeddings: {str(e)}") from e