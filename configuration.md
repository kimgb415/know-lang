# Configuration Guide

KnowLang uses [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) for configuration management. Settings can be provided through environment variables, `.env` files, or programmatically.

## Quick Start

1. Copy the example configuration:
```bash
cp .env.example .env
```

2. Modify settings as needed in `.env`

## Core Settings

### LLM Settings
```env
# Default is Ollama with llama3.2
LLM__MODEL_NAME=llama3.2
LLM__MODEL_PROVIDER=ollama
LLM__API_KEY=your_api_key  # Required for providers like OpenAI
```

Supported providers:
- `ollama`: Local models through Ollama
- `openai`: OpenAI models (requires API key)
- `anthropic`: Anthropic models (requires API key)

### Embedding Settings
```env
# Default is Ollama with mxbai-embed-large
EMBEDDING__MODEL_NAME=mxbai-embed-large
EMBEDDING__MODEL_PROVIDER=ollama
EMBEDDING__API_KEY=your_api_key  # Required for providers like OpenAI
```

### Database Settings
```env
# ChromaDB configuration
DB__PERSIST_DIRECTORY=./chromadb/mycode
DB__COLLECTION_NAME=code
DB__CODEBASE_DIRECTORY=./
```

### Parser Settings
```env
# Language support and file patterns
PARSER__LANGUAGES='{"python": {"enabled": true, "file_extensions": [".py"]}}'
PARSER__PATH_PATTERNS='{"include": ["**/*"], "exclude": ["**/venv/**", "**/.git/**"]}'
```

### Chat Interface Settings
```env
CHAT__MAX_CONTEXT_CHUNKS=5
CHAT__SIMILARITY_THRESHOLD=0.7
CHAT__INTERFACE_TITLE='Code Repository Q&A Assistant'
```

## Advanced Configuration

### Using Multiple Models

You can configure different models for different purposes:
```env
# Main LLM for responses
LLM__MODEL_NAME=llama3.2
LLM__MODEL_PROVIDER=ollama

# Evaluation model
EVALUATOR__MODEL_NAME=gpt-4
EVALUATOR__MODEL_PROVIDER=openai

# Embedding model
EMBEDDING__MODEL_NAME=mxbai-embed-large
EMBEDDING__MODEL_PROVIDER=ollama
```

### Reranker Configuration
```env
RERANKER__ENABLED=true
RERANKER__MODEL_NAME=rerank-2
RERANKER__MODEL_PROVIDER=voyage
RERANKER__TOP_K=4
```

### Analytics Integration
```env
CHAT_ANALYTICS__ENABLED=true
CHAT_ANALYTICS__PROVIDER=mixpanel
CHAT_ANALYTICS__API_KEY=your_api_key
```


## Further Reading

- For detailed settings configuration options, see [pydantic-settings documentation](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- For model-specific configuration, see provider documentation:
  - [Ollama Models](https://ollama.ai/library)
  - [OpenAI Models](https://platform.openai.com/docs/models)
  - [Anthropic Models](https://www.anthropic.com/models)