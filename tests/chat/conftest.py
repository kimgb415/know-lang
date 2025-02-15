
from unittest.mock import Mock
import pytest
from knowlang.configs.config import AppConfig, LLMConfig, RerankerConfig
from knowlang.core.types import ModelProvider


@pytest.fixture
def mock_collection():
    collection = Mock()
    collection.query = Mock(return_value={
        'documents': [['mock code chunk 1', 'mock code chunk 2']],
        'metadatas': [[
            {'file_path': 'test.py', 'start_line': 1, 'end_line': 10},
            {'file_path': 'test2.py', 'start_line': 5, 'end_line': 15}
        ]],
        'distances': [[0.1, 0.2]]
    })
    return collection

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