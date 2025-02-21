from typing import Callable, Dict, List, Optional

import ollama
import openai
import voyageai

from knowlang.configs.config import ModelProvider
from knowlang.models.types import EmbeddingInputType, EmbeddingVector

# Global registry for provider functions
PROVIDER_REGISTRY: Dict[ModelProvider, Callable[[List[str], str, Optional[EmbeddingInputType]], List[EmbeddingVector]]] = {}

def register_provider(provider: ModelProvider):
    """Decorator to register a provider function."""
    def decorator(func: Callable[[List[str], str, Optional[EmbeddingInputType]], List[EmbeddingVector]]):
        PROVIDER_REGISTRY[provider] = func
        return func
    return decorator

@register_provider(ModelProvider.OLLAMA)
def _process_ollama_batch(inputs: List[str], model_name: str, _: Optional[EmbeddingInputType] = None) -> List[EmbeddingVector]:
    return ollama.embed(model=model_name, input=inputs)['embeddings']

@register_provider(ModelProvider.OPENAI)
def _process_openai_batch(inputs: List[str], model_name: str, _: Optional[EmbeddingInputType] = None) -> List[EmbeddingVector]:
    response = openai.embeddings.create(input=inputs, model=model_name)
    return [item.embedding for item in response.data]

@register_provider(ModelProvider.VOYAGE)
def _process_voyage_batch(inputs: List[str], model_name: str, input_type: Optional[EmbeddingInputType]) -> List[EmbeddingVector]:
    client = voyageai.Client()
    embeddings_obj = client.embed(model=model_name, texts=inputs, input_type=input_type.value)
    return embeddings_obj.embeddings