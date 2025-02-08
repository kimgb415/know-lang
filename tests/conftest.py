import pytest
import tempfile
import git
from pathlib import Path
from typing import Dict
from knowlang.configs.config import AppConfig, ParserConfig, LanguageConfig
from knowlang.parser.languages.python.parser import PythonParser
from tests.test_data.python_files import TEST_FILES

@pytest.fixture
def test_config() -> AppConfig:
    """Provides test configuration"""
    return AppConfig(
        parser=ParserConfig(
            languages={
                "python": LanguageConfig(
                    file_extensions=[".py"],
                    tree_sitter_language="python",
                    max_file_size=1_000_000,
                    chunk_types=["class_definition", "function_definition"]
                )
            }
        )
    )

@pytest.fixture
def temp_repo():
    """Create a temporary git repository with sample Python files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        repo = git.Repo.init(temp_dir)
        
        for filename, content in TEST_FILES.items():
            file_path = Path(temp_dir) / filename
            file_path.write_text(content)
            repo.index.add([str(file_path)])
        
        repo.index.commit("Initial commit")
        
        yield temp_dir

@pytest.fixture
def python_parser(test_config):
    """Provides initialized Python parser"""
    parser = PythonParser(test_config)
    parser.setup()
    return parser