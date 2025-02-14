import pytest
from pathlib import Path
from knowlang.configs.config import AppConfig
from knowlang.core.types import BaseChunkType
from tests.test_data.python_files import (
    TEST_FILES,
    INVALID_SYNTAX, 
    SIMPLE_FILE_EXPECTATIONS,
    NESTED_CLASS_EXPECTATIONS,
    COMPLEX_FILE_EXPECTATIONS
)
from knowlang.core.types import CodeChunk
from knowlang.parser.languages.python.parser import PythonParser
from typing import List
import tempfile


@pytest.fixture
def python_parser(test_config):
    """Provides initialized Python parser"""
    parser = PythonParser(test_config)
    parser.setup()
    return parser

def find_chunk_by_criteria(chunks: List[CodeChunk], **criteria) -> CodeChunk:
    """Helper function to find a chunk matching given criteria"""
    for chunk in chunks:
        if all(getattr(chunk, k) == v for k, v in criteria.items()):
            return chunk
    return None

def verify_chunk_matches_expectation(
    chunk: CodeChunk,
    expected_name: str,
    expected_docstring: str,
    expected_content_snippet: str
) -> bool:
    """Verify that a chunk matches expected values"""
    return (
        chunk.name == expected_name and
        expected_content_snippet in chunk.content and
        chunk.docstring is not None and
        expected_docstring in chunk.docstring
    )


class TestPythonParser:
    """Test suite for PythonParser"""

    def test_parser_initialization(self, python_parser: PythonParser):
        """Test parser initialization"""
        assert python_parser.parser is not None
        assert python_parser.language is not None

    def test_simple_file_parsing(self, python_parser: PythonParser, test_config: AppConfig):
        """Test parsing a simple Python file with function and class"""
        chunks = python_parser.parse_file(test_config.db.codebase_directory / "simple.py")

        # Test function
        function_chunk = find_chunk_by_criteria(
            chunks,
            type=BaseChunkType.FUNCTION,
            name="hello_world"
        )
        assert function_chunk is not None
        expected = SIMPLE_FILE_EXPECTATIONS['hello_world']
        assert verify_chunk_matches_expectation(
            function_chunk,
            expected.name,
            expected.docstring,
            expected.content_snippet
        )

        # Test class
        class_chunk = find_chunk_by_criteria(
            chunks,
            type=BaseChunkType.CLASS,
            name="SimpleClass"
        )
        assert class_chunk is not None
        expected = SIMPLE_FILE_EXPECTATIONS['SimpleClass']
        assert verify_chunk_matches_expectation(
            class_chunk,
            expected.name,
            expected.docstring,
            expected.content_snippet
        )

    def test_complex_file_parsing(self, python_parser: PythonParser, test_config: AppConfig):
        """Test parsing a complex Python file"""
        chunks = python_parser.parse_file(Path(test_config.db.codebase_directory) / "complex.py")
        
        # Test complex function
        complex_func = find_chunk_by_criteria(
            chunks,
            type=BaseChunkType.FUNCTION,
            name="complex_function"
        )
        assert complex_func is not None
        expected = COMPLEX_FILE_EXPECTATIONS['complex_function']
        assert verify_chunk_matches_expectation(
            complex_func,
            expected.name,
            expected.docstring,
            expected.content_snippet
        )
        
        # Test complex class
        complex_class = find_chunk_by_criteria(
            chunks,
            type=BaseChunkType.CLASS,
            name="ComplexClass"
        )
        assert complex_class is not None
        expected = COMPLEX_FILE_EXPECTATIONS['ComplexClass']
        assert verify_chunk_matches_expectation(
            complex_class,
            expected.name,
            expected.docstring,
            expected.content_snippet
        )

    def test_error_handling(self, python_parser: PythonParser, test_config: AppConfig):
        """Test error handling for various error cases"""
        # Test invalid syntax
        invalid_file = Path(test_config.db.codebase_directory) / "invalid.py"
        invalid_file.write_text(INVALID_SYNTAX)
        chunks = python_parser.parse_file(invalid_file)
        assert chunks == []
        
        # Test non-existent file
        nonexistent = Path(test_config.db.codebase_directory) / "nonexistent.py"
        chunks = python_parser.parse_file(nonexistent)
        assert chunks == []
        
        # Test non-Python file
        non_python = Path(test_config.db.codebase_directory) / "readme.md"
        non_python.write_text("# README")
        chunks = python_parser.parse_file(non_python)
        assert chunks == []

    def test_file_size_limits(self, python_parser: PythonParser, test_config: AppConfig):
        """Test file size limit enforcement"""
        large_file = Path(test_config.db.codebase_directory) / "large.py"
        # Create a file larger than the limit
        large_file.write_text("x = 1\n" * 1_000_000)
        
        chunks = python_parser.parse_file(large_file)
        assert chunks == []

    @pytest.mark.parametrize("test_file", TEST_FILES.keys())
    def test_supported_extensions(self, python_parser: PythonParser, test_file: str):
        """Test file extension support"""
        assert any(test_file.endswith(ext) for ext in python_parser.language_config.file_extensions)
