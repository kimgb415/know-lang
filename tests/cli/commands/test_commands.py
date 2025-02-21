"""Unit tests for CLI command implementations."""
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from knowlang.cli.commands.chat import chat_command, create_config
from knowlang.cli.commands.parse import parse_command
from knowlang.cli.types import ChatCommandArgs, ParseCommandArgs
from knowlang.configs.config import AppConfig
from knowlang.core.types import CodeChunk
from knowlang.indexing.state_store.base import FileChange, StateChangeType
from knowlang.vector_stores import VectorStoreError
from knowlang.vector_stores.mock import MockVectorStore


@pytest.fixture
def mock_parser_factory():
    with patch('knowlang.cli.commands.parse.CodeParserFactory') as factory:
        # Set up the mock parser behavior
        mock_parser = Mock()
        mock_parser.parse_file.return_value = [MagicMock(spec=CodeChunk)]
        factory.return_value.get_parser.return_value = mock_parser
        yield factory

@pytest.fixture
def mock_codebase_manager():
    with patch('knowlang.cli.commands.parse.CodebaseManager') as manager:
        mock_instance = AsyncMock()
        mock_instance.get_current_files = AsyncMock(return_value={Path("test.py")})
        manager.return_value = mock_instance
        yield manager

@pytest.fixture
def mock_state_manager():
    with patch('knowlang.cli.commands.parse.StateManager') as manager:
        mock_instance = AsyncMock()
        # Mock the state_store attribute
        mock_instance.state_store = AsyncMock()
        mock_instance.state_store.detect_changes = AsyncMock()
        # Set up default behavior to return a list with one FileChange
        file_change = FileChange(path=Path("test.py"), change_type=StateChangeType.ADDED)
        mock_instance.state_store.detect_changes.return_value = [file_change]
        manager.return_value = mock_instance
        yield manager

@pytest.fixture
def mock_incremental_updater():
    with patch('knowlang.cli.commands.parse.IncrementalUpdater') as updater:
        mock_instance = AsyncMock()
        mock_instance.update_codebase = AsyncMock()
        updater.return_value = mock_instance
        yield updater

@pytest.fixture
def mock_formatter():
    with patch('knowlang.cli.commands.parse.get_formatter') as formatter_func:
        mock_formatter = Mock()
        mock_formatter.display_chunks = Mock()
        formatter_func.return_value = mock_formatter
        yield formatter_func

@pytest.fixture
def mock_progress_tracker():
    with patch('knowlang.cli.commands.parse.ProgressTracker') as tracker:
        mock_instance = MagicMock()
        # Mock the context manager behavior
        mock_instance.progress.return_value.__enter__.return_value = None
        mock_instance.progress.return_value.__exit__.return_value = None
        tracker.return_value = mock_instance
        yield tracker

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
    async def test_parse_codebase(
        self,
        mock_parser_factory,
        mock_codebase_manager,
        mock_state_manager,
        mock_incremental_updater,
        mock_formatter,
        mock_progress_tracker,
        tmp_path
    ):
        """Test parsing a codebase with the new implementation."""
        # Setup
        codebase_dir = tmp_path / "codebase"
        codebase_dir.mkdir()
        
        args = ParseCommandArgs(
            verbose=False,
            config=None,
            path=codebase_dir,
            output="table",
            command="parse"
        )
        
        # Execute
        await parse_command(args)
        
        # Assert CodebaseManager was created and used
        mock_codebase_manager.assert_called_once()
        mock_codebase_manager.return_value.get_current_files.assert_called_once()
        
        # Assert StateManager was created and used
        mock_state_manager.assert_called_once()
        mock_state_manager.return_value.state_store.detect_changes.assert_called_once()
        
        # Assert parser was used for changed files
        mock_parser_factory.return_value.get_parser.assert_called_once()
        mock_parser_factory.return_value.get_parser.return_value.parse_file.assert_called_once()
        
        # Assert IncrementalUpdater was created and used
        mock_incremental_updater.assert_called_once()
        mock_incremental_updater.return_value.update_codebase.assert_called_once()
        
        # Assert formatter was used
        mock_formatter.assert_called_once_with("table")
        mock_formatter.return_value.display_chunks.assert_called_once()
        
        # Assert progress tracker was used
        mock_progress_tracker.assert_called_once()

    @pytest.mark.asyncio
    async def test_parse_empty_codebase(
        self,
        mock_parser_factory,
        mock_codebase_manager,
        mock_state_manager,
        mock_incremental_updater,
        mock_formatter,
        mock_progress_tracker,
        tmp_path
    ):
        """Test parsing an empty codebase with no changes."""
        # Setup
        codebase_dir = tmp_path / "empty_codebase"
        codebase_dir.mkdir()
        
        args = ParseCommandArgs(
            verbose=False,
            config=None,
            path=codebase_dir,
            output="table",
            command="parse"
        )
        
        # Mock empty codebase
        mock_codebase_manager.return_value.get_current_files.return_value = set()
        mock_state_manager.return_value.state_store.detect_changes.return_value = []
        
        # Execute
        await parse_command(args)
        
        # Assert CodebaseManager and StateManager were used
        mock_codebase_manager.assert_called_once()
        mock_state_manager.assert_called_once()
        
        # Assert IncrementalUpdater was created and used with empty list
        mock_incremental_updater.assert_called_once()
        mock_incremental_updater.return_value.update_codebase.assert_called_once_with(
            chunks=[], file_changes=[]
        )
        
        # Assert formatter was not used for chunks (no chunks found)
        mock_formatter.return_value.display_chunks.assert_not_called()

    @pytest.mark.asyncio
    async def test_parse_with_multiple_file_changes(
        self,
        mock_parser_factory,
        mock_codebase_manager,
        mock_state_manager,
        mock_incremental_updater,
        mock_formatter,
        mock_progress_tracker,
        tmp_path
    ):
        """Test parsing a codebase with multiple file changes."""
        # Setup
        codebase_dir = tmp_path / "multi_file_codebase"
        codebase_dir.mkdir()
        
        args = ParseCommandArgs(
            verbose=False,
            config=None,
            path=codebase_dir,
            output="table",
            command="parse"
        )
        
        # Mock multiple files
        mock_codebase_manager.return_value.get_current_files.return_value = {
            Path("file1.py"), Path("file2.py"), Path("file3.py")
        }
        
        # Mock multiple file changes
        file_changes = [
            FileChange(path=Path("file1.py"), change_type=StateChangeType.ADDED),
            FileChange(path=Path("file2.py"), change_type=StateChangeType.MODIFIED),
            FileChange(path=Path("file3.py"), change_type=StateChangeType.DELETED)
        ]
        mock_state_manager.return_value.state_store.detect_changes.return_value = file_changes
        
        # Mock parser returning different chunks for different files
        def get_mock_parser(file_path):
            parser = Mock()
            num_chunks = 1 if "file1" in str(file_path) else 2 if "file2" in str(file_path) else 0
            parser.parse_file.return_value = [MagicMock(spec=CodeChunk) for _ in range(num_chunks)]
            return parser
        
        mock_parser_factory.return_value.get_parser.side_effect = get_mock_parser
        
        # Execute
        await parse_command(args)
        
        # Assert parser was called for each file (except deleted ones)
        assert mock_parser_factory.return_value.get_parser.call_count == 2
        
        # Assert IncrementalUpdater was called with correct number of chunks (1+2=3)
        # Since the parse_file method is actually mocked and not called for file3 (which is deleted)
        assert len(mock_incremental_updater.return_value.update_codebase.call_args[1]['chunks']) == 3
        
        # Assert file changes were passed correctly
        assert mock_incremental_updater.return_value.update_codebase.call_args[1]['file_changes'] == file_changes



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