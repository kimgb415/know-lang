# LLM Configuration
# Model settings for the main language model
LLM__MODEL_NAME=llama3.2
LLM__MODEL_PROVIDER=ollama
LLM__API_KEY=your_api_key

# Evaluator Configuration
# Settings for the model evaluation
EVALUATOR__MODEL_NAME=llama3.2
EVALUATOR__MODEL_PROVIDER=ollama
EVALUATOR__API_KEY=your_api_key
EVALUATOR__EVALUATION_ROUNDS=1

# Reranker Configuration
# Settings for result reranking
RERANKER__ENABLED=false
RERANKER__MODEL_NAME=rerank-2
RERANKER__MODEL_PROVIDER=voyage
RERANKER__API_KEY=your_api_key
RERANKER__TOP_K=4
RERANKER__RELEVANCE_THRESHOLD=0.5

# Database Configuration
# ChromaDB and codebase settings
DB__PERSIST_DIRECTORY=./chromadb/mycode
DB__COLLECTION_NAME=code
DB__CODEBASE_DIRECTORY=./

# Parser Configuration
# Settings for code parsing and file patterns
PARSER__LANGUAGES='{"python": {"enabled": true, "file_extensions": [".py"], "tree_sitter_language": "python", "chunk_types": ["class_definition", "function_definition"], "max_file_size": 1000000}}'
PARSER__PATH_PATTERNS='{"include": ["**/*"], "exclude": ["**/venv/**", "**/.git/**", "**/__pycache__/**", "**/tests/**"]}'

# Chat Configuration
# Settings for the chat interface and context handling
CHAT__MAX_CONTEXT_CHUNKS=5
CHAT__SIMILARITY_THRESHOLD=0.7
CHAT__INTERFACE_TITLE='Code Repository Q&A Assistant'
CHAT__INTERFACE_DESCRIPTION="Ask questions about the codebase and I'll help you understand it!"
CHAT__MAX_LENGTH_PER_CHUNK=8000

# Embedding Configuration
# Settings for text embedding generation
EMBEDDING__MODEL_NAME=mxbai-embed-large
EMBEDDING__MODEL_PROVIDER=ollama
EMBEDDING__API_KEY=your_api_key

# Chat Analytics Configuration
# Settings for analytics tracking
CHAT_ANALYTICS__ENABLED=false
CHAT_ANALYTICS__PROVIDER=mixpanel
CHAT_ANALYTICS__API_KEY=your_api_key