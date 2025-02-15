
import pytest
from knowlang.configs.config import AppConfig, LLMConfig, RerankerConfig
from knowlang.core.types import ModelProvider
from knowlang.vector_stores.mock import MockVectorStore

@pytest.fixture
def mock_config():
    return AppConfig(
        llm = LLMConfig(
            model_provider=ModelProvider.TESTING
        ),
        reranker=RerankerConfig(
            enabled=True,
            model_provider=ModelProvider.VOYAGE,
            relevance_threshold=0.7,
            api_key="test_key"
        )
    )

@pytest.fixture
def mock_vector_store():
    """Fixture providing a clean mock vector store for each test"""
    store = MockVectorStore()
    yield store
    store.reset()


@pytest.fixture
def populated_mock_store():
    """Fixture providing a mock store with some test data"""
    store = MockVectorStore()
    
    # Add some test documents
    docs = [
        "def test_function(): pass",
        "class TestClass: pass",
        "import test_module"
    ]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]
    metadatas = [
        {"file_path": "test1.py", "start_line": 1, "end_line": 2},
        {"file_path": "test2.py", "start_line": 3, "end_line": 4},
        {"file_path": "test3.py", "start_line": 5, "end_line": 6}
    ]
    
    async def init_store():
        await store.add_documents(docs, embeddings, metadatas)
    
    import asyncio
    asyncio.run(init_store())
    
    yield store
    store.reset()