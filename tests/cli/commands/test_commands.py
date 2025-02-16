"""Unit tests for CLI command implementations."""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from knowlang.cli.commands.parse import parse_command
from knowlang.cli.commands.chat import chat_command, create_config
from knowlang.cli.types import ParseCommandArgs, ChatCommandArgs
from knowlang.configs.config import AppConfig
from knowlang.vector_stores.base import VectorStoreError
from knowlang.vector_stores.mock import MockVectorStore

@pytest.fixture
def mock_parser_factory():
    with patch('knowlang.cli.commands.parse.CodeParserFactory') as factory:
        yield factory

@pytest.fixture
def mock_git_provider():
    with patch('knowlang.cli.commands.parse.GitProvider') as provider:
        yield provider

@pytest.fixture
def mock_filesystem_provider():
    with patch('knowlang.cli.commands.parse.FilesystemProvider') as provider:
        yield provider

@pytest.fixture
def mock_summarizer():
    with patch('knowlang.cli.commands.parse.CodeSummarizer') as summarizer:
        mock_instance = Mock()
        mock_instance.process_chunks = AsyncMock()
        summarizer.return_value = mock_instance
        yield summarizer

@pytest.fixture
def mock_formatter():
    with patch('knowlang.cli.commands.parse.get_formatter') as formatter:
        yield formatter

@pytest.fixture
def mock_vector_store():
    """Mock vector store instance"""
    store = MockVectorStore()
    return store

@pytest.fixture
def mock_vector_store_factory(mock_vector_store: MockVectorStore):
    """Mock vector store factory"""
    with patch('knowlang.cli.commands.chat.VectorStoreFactory') as factory:
        factory.get.return_value = mock_vector_store
        yield factory


@pytest.fixture
def mock_chatbot():
    with patch('knowlang.cli.commands.chat.create_chatbot') as chatbot:
        mock_demo = Mock()
        mock_demo.launch = Mock()
        chatbot.return_value = mock_demo
        yield chatbot

class TestParseCommand:
    @pytest.mark.asyncio
    async def test_parse_git_repository(
        self,
        mock_parser_factory,
        mock_git_provider,
        mock_filesystem_provider,
        mock_summarizer,
        mock_formatter,
        tmp_path
    ):
        """Test parsing a Git repository."""
        # Setup
        git_dir = tmp_path / "repo"
        git_dir.mkdir()
        (git_dir / ".git").mkdir()
        
        args = ParseCommandArgs(
            verbose=False,
            config=None,
            path=git_dir,
            output="table",
            command="parse"
        )
        
        mock_provider_instance = Mock()
        mock_provider_instance.get_files.return_value = [Path("test.py")]
        mock_git_provider.return_value = mock_provider_instance
        
        mock_parser = Mock()
        mock_parser.parse_file.return_value = ["chunk1", "chunk2"]
        mock_parser_factory.return_value.get_parser.return_value = mock_parser
        
        # Execute
        await parse_command(args)
        
        # Assert
        mock_git_provider.assert_called_once()
        mock_filesystem_provider.assert_not_called()
        mock_provider_instance.get_files.assert_called_once()
        mock_parser.parse_file.assert_called_once()
        mock_formatter.return_value.display_chunks.assert_called_once_with(["chunk1", "chunk2"])
        mock_summarizer.return_value.process_chunks.assert_called_once_with(["chunk1", "chunk2"])

    @pytest.mark.asyncio
    async def test_parse_filesystem(
        self,
        mock_parser_factory,
        mock_git_provider,
        mock_filesystem_provider,
        mock_summarizer,
        mock_formatter,
        tmp_path
    ):
        """Test parsing a regular directory."""
        args = ParseCommandArgs(
            verbose=False,
            config=None,
            path=tmp_path,
            output="table",
            command="parse"
        )
        
        mock_provider_instance = Mock()
        mock_provider_instance.get_files.return_value = [Path("test.py")]
        mock_filesystem_provider.return_value = mock_provider_instance
        
        mock_parser = Mock()
        mock_parser.parse_file.return_value = ["chunk1"]
        mock_parser_factory.return_value.get_parser.return_value = mock_parser
        
        await parse_command(args)
        
        mock_git_provider.assert_not_called()
        mock_filesystem_provider.assert_called_once()
        mock_provider_instance.get_files.assert_called_once()
        mock_parser.parse_file.assert_called_once()
        mock_formatter.return_value.display_chunks.assert_called_once_with(["chunk1"])

class TestChatCommand:
    @pytest.mark.asyncio
    async def test_chat_with_working_vector_store(
        self, 
        mock_vector_store, 
        mock_vector_store_factory,
        mock_chatbot
    ):
        """Test chat command with a working vector store."""
        args = ChatCommandArgs(
            verbose=False,
            config=None,
            command="chat",
            port=None,
            share=False,
            server_port=None,
            server_name=None
        )
        
        # Setup mock vector store
        mock_vector_store_factory.get.return_value = mock_vector_store
        
        await chat_command(args)
        
        # Verify vector store creation
        mock_vector_store_factory.get.assert_called_once()
        
        # Verify chatbot creation and launch
        mock_chatbot.assert_called_once()
        mock_chatbot.return_value.launch.assert_called_once_with(
            server_port=None,
            server_name="127.0.0.1",
            share=False
        )

    @pytest.mark.asyncio
    async def test_chat_with_vector_store_error(
        self,
        mock_vector_store_factory,
        mock_chatbot
    ):
        """Test chat command when vector store initialization fails."""
        args = ChatCommandArgs(
            verbose=False,
            config=None,
            command="chat",
            port=None,
            share=False,
            server_port=None,
            server_name=None
        )
        
        # Setup mock vector store to raise error
        mock_vector_store_factory.get.side_effect = VectorStoreError("Store not found")
        
        await chat_command(args)
        
        # Verify vector store creation attempt
        mock_vector_store_factory.get.assert_called_once()
        
        # Verify chatbot was not created
        mock_chatbot.assert_not_called()

    def test_create_config_with_file(self, tmp_path):
        """Test config creation from file."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"db": {"persist_directory": "test_dir"}}')
        
        args = ChatCommandArgs(
            verbose=False,
            config=config_file,
            command="chat",
            port=None,
            share=False,
            server_port=None,
            server_name=None
        )
        
        config = create_config(args)
        assert isinstance(config, AppConfig)
        assert config.db.persist_directory == Path("test_dir")

    def test_create_config_without_file(self):
        """Test config creation with defaults."""
        args = ChatCommandArgs(
            verbose=False,
            config=None,
            command="chat",
            port=None,
            share=False,
            server_port=None,
            server_name=None
        )
        
        config = create_config(args)
        assert isinstance(config, AppConfig) 