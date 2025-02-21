import tempfile
from pathlib import Path
from typing import Generator

import pytest

from knowlang.configs import AppConfig, DBConfig, LanguageConfig, ParserConfig


@pytest.fixture
def test_config() -> Generator[AppConfig, None, None]:
    """Provides test configuration"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield AppConfig(
            parser=ParserConfig(
                languages={
                    "cpp": LanguageConfig(
                        file_extensions=[".cpp", ".hpp", ".h"],
                        tree_sitter_language="cpp",
                        max_file_size=1_000_000,
                        chunk_types=["class_definition", "function_definition"]
                    )
                }
            ),
            db=DBConfig(
                codebase_directory=Path(temp_dir)
            )
        )