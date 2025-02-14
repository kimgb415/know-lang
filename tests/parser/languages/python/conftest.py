from typing import Generator
import pytest
import tempfile
import git
from pathlib import Path
from knowlang.configs.config import AppConfig, ParserConfig, LanguageConfig, DBConfig
from tests.test_data.python_files import TEST_FILES

@pytest.fixture
def test_config() -> Generator[AppConfig, None, None]:
    """Provides test configuration"""
    with tempfile.TemporaryDirectory() as temp_dir:
        repo = git.Repo.init(temp_dir)
        
        for filename, content in TEST_FILES.items():
            file_path = Path(temp_dir) / filename
            file_path.write_text(content)
            repo.index.add([str(file_path)])
        
        repo.index.commit("Initial commit")
        
        yield AppConfig(
            parser=ParserConfig(
                languages={
                    "python": LanguageConfig(
                        file_extensions=[".py"],
                        tree_sitter_language="python",
                        max_file_size=1_000_000,
                        chunk_types=["class_definition", "function_definition"]
                    )
                }
            ),
            db=DBConfig(
                codebase_directory=Path(temp_dir)
            )
        )
