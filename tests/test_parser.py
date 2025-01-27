from know_lang_bot.code_parser.parser import CodeChunk, CodeParser, ChunkType
from pathlib import Path
from tests.test_constants import (
    SIMPLE_FILE_EXPECTATIONS,
    NESTED_CLASS_EXPECTATIONS,
    COMPLEX_FILE_EXPECTATIONS,
    INVALID_SYNTAX,
    TEST_FILES,
)
import pytest
import tempfile
import git


@pytest.fixture
def temp_repo():
    """Create a temporary git repository with sample Python files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize git repo
        repo = git.Repo.init(temp_dir)
        
        # Create sample Python files
        for filename, content in TEST_FILES.items():
            file_path = Path(temp_dir) / filename
            file_path.write_text(content)
            repo.index.add([str(file_path)])
        
        repo.index.commit("Initial commit")
        
        yield temp_dir

def find_chunk_by_criteria(chunks: list[CodeChunk], **criteria) -> CodeChunk:
    """Helper function to find a chunk matching given criteria"""
    for chunk in chunks:
        if all(getattr(chunk, k) == v for k, v in criteria.items()):
            return chunk
    return None

def test_init_parser(temp_repo):
    """Test parser initialization"""
    parser = CodeParser(temp_repo)
    assert parser.repo_path == Path(temp_repo)
    assert parser.language is not None
    assert parser.parser is not None

def test_parse_simple_file(temp_repo):
    """Test parsing a simple Python file with function and class"""
    parser = CodeParser(temp_repo)
    chunks = parser.parse_file(Path(temp_repo) / "simple.py")

    # Test function
    function_chunk = find_chunk_by_criteria(chunks, type=ChunkType.FUNCTION, name="hello_world")
    assert function_chunk is not None
    expected = SIMPLE_FILE_EXPECTATIONS['hello_world']
    assert expected.content_snippet in function_chunk.content
    assert function_chunk.docstring is not None
    assert function_chunk.docstring in expected.docstring

    # Test class
    class_chunk = find_chunk_by_criteria(chunks, type=ChunkType.CLASS, name="SimpleClass")
    assert class_chunk is not None
    expected = SIMPLE_FILE_EXPECTATIONS['SimpleClass']
    assert expected.content_snippet in class_chunk.content
    assert class_chunk.docstring is not None
    assert class_chunk.docstring in expected.docstring


def test_parse_nested_classes(temp_repo):
    """Test parsing nested class definitions"""
    parser = CodeParser(temp_repo)
    chunks = parser.parse_file(Path(temp_repo) / "nested.py")
    
    # Test outer class
    outer_class = find_chunk_by_criteria(chunks, type=ChunkType.CLASS, name="OuterClass")
    assert outer_class is not None
    expected = NESTED_CLASS_EXPECTATIONS['OuterClass']
    assert expected.content_snippet in outer_class.content
    assert outer_class.docstring is not None
    assert outer_class.docstring in expected.docstring

    # Verify inner class: Not implemented yet
    pass

def test_parse_complex_file(temp_repo):
    """Test parsing a complex Python file"""
    parser = CodeParser(temp_repo)
    chunks = parser.parse_file(Path(temp_repo) / "complex.py")
    
    # Test function with type hints
    complex_func = find_chunk_by_criteria(
        chunks, 
        type=ChunkType.FUNCTION,
        name="complex_function"
    )
    assert complex_func is not None
    expected = COMPLEX_FILE_EXPECTATIONS['complex_function']
    assert expected.content_snippet in complex_func.content
    assert complex_func.docstring is not None
    assert complex_func.docstring in expected.docstring
    
    # Test complex class
    complex_class = find_chunk_by_criteria(
        chunks,
        type=ChunkType.CLASS,
        name="ComplexClass"
    )
    assert complex_class is not None
    expected = COMPLEX_FILE_EXPECTATIONS['ComplexClass']
    assert expected.content_snippet in complex_class.content
    assert complex_class.docstring is not None
    assert complex_class.docstring in expected.docstring 


def test_parse_repository(temp_repo):
    """Test parsing entire repository"""
    parser = CodeParser(temp_repo)
    chunks = parser.parse_repository()
    
    file_paths = {chunk.file_path for chunk in chunks}
    assert len(file_paths) == 3
    
    # Verify we can find chunks from each test file
    for filename in TEST_FILES.keys():
        file_chunks = [c for c in chunks if Path(c.file_path).name == filename]
        assert len(file_chunks) > 0

def test_error_handling(temp_repo):
    """Test error handling for invalid files"""
    parser = CodeParser(temp_repo)
    
    # Test invalid syntax
    invalid_file = Path(temp_repo) / "invalid.py"
    invalid_file.write_text(INVALID_SYNTAX)
    chunks = parser.parse_file(invalid_file)
    assert chunks == []
    
    # Test non-existent file
    nonexistent = Path(temp_repo) / "nonexistent.py"
    chunks = parser.parse_file(nonexistent)
    assert chunks == []

def test_non_python_files(temp_repo):
    """Test handling of non-Python files"""
    parser = CodeParser(temp_repo)
    
    # Create a non-Python file
    non_python = Path(temp_repo) / "readme.md"
    non_python.write_text("# README")
    
    # Should skip non-Python files
    chunks = parser.parse_file(non_python)
    assert chunks == []