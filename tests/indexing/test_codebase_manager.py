import pytest
import tempfile
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Set

from knowlang.indexing.codebase_manager import CodebaseManager
from knowlang.configs.config import AppConfig, ParserConfig, PathPatterns

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname).resolve()

@pytest.fixture
def temp_config(temp_dir: Path):
    config = AppConfig()
    config.parser = ParserConfig(
        path_patterns=PathPatterns(
            include=["*.py"],
            exclude=["*.pyc", "__pycache__/*"]
        )
    )
    config.db.codebase_directory = temp_dir

    return config


@pytest.fixture
def codebase_manager(temp_config: AppConfig):
    """Create CodebaseManager with default test config"""
    return CodebaseManager(temp_config)

def create_temp_file(directory: Path, filename: str, content: str = "") -> Path:
    """Helper to create a temporary file with content"""
    file_path = directory / filename
    file_path.write_text(content)
    return file_path

@pytest.mark.asyncio
async def test_get_current_files_basic(codebase_manager: CodebaseManager, temp_dir: Path):
    """Test basic file scanning with default patterns"""
    # Create test files
    py_file = create_temp_file(temp_dir, "test.py", "print('hello')")
    pyc_file = create_temp_file(temp_dir, "test.pyc", "compiled")
    txt_file = create_temp_file(temp_dir, "test.txt", "text")
    
    # Get current files
    files = await codebase_manager.get_current_files()
    
    # Verify only .py files are included
    assert len(files) == 1
    assert temp_dir / "test.py" in files
    assert temp_dir / "test.pyc" not in files
    assert temp_dir / "test.txt" not in files

@pytest.mark.asyncio
async def test_get_current_files_nested(codebase_manager: CodebaseManager, temp_dir: Path):
    """Test file scanning with nested directories"""
    # Create nested directory structure
    subdir = temp_dir / "subdir"
    subdir.mkdir()
    nested = subdir / "nested"
    nested.mkdir()
    
    # Create files in different directories
    create_temp_file(temp_dir, "root.py", "root")
    create_temp_file(subdir, "sub.py", "sub")
    create_temp_file(nested, "nested.py", "nested")
    
    # Get current files
    files = await codebase_manager.get_current_files()
    
    # Verify all .py files are found
    assert len(files) == 3
    assert temp_dir / "root.py" in files
    assert temp_dir / "subdir" / "sub.py" in files
    assert temp_dir / "subdir" / "nested" / "nested.py" in files


@pytest.mark.asyncio
async def test_create_file_state(codebase_manager: CodebaseManager, temp_dir: Path):
    """Test creation of FileState objects"""
    # Create test file
    test_file = create_temp_file(temp_dir, "test.py", "test content")
    chunk_ids = {"chunk1", "chunk2"}
    
    # Create file state
    state = await codebase_manager.create_file_state(test_file, chunk_ids)
    
    # Verify state properties
    assert state.file_path in str(test_file)
    assert isinstance(state.last_modified, datetime)
    assert state.chunk_ids == chunk_ids

@pytest.mark.asyncio
async def test_file_state_timestamp(codebase_manager: CodebaseManager, temp_dir: Path):
    """Test that file state timestamps accurately reflect modifications"""
    # Create initial file
    test_file = create_temp_file(temp_dir, "test.py", "initial content")
    initial_state = await codebase_manager.create_file_state(test_file, {"chunk1"})
    initial_time = initial_state.last_modified
    
    # Wait a moment to ensure timestamp would change
    import asyncio
    await asyncio.sleep(0.1)
    
    # Modify file
    test_file.write_text("modified content")
    modified_state = await codebase_manager.create_file_state(test_file, {"chunk2"})
    
    # Verify timestamp changed
    assert modified_state.last_modified > initial_time
    assert modified_state.file_hash != initial_state.file_hash