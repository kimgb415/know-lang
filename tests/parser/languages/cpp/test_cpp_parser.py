import pytest
from pathlib import Path
from typing import List
from knowlang.core.types import BaseChunkType, CodeChunk
from knowlang.parser.languages.cpp.parser import CppParser
from knowlang.configs.config import AppConfig

@pytest.fixture
def cpp_parser(test_config):
    """Provides initialized cpp parser"""
    parser = CppParser(test_config)
    parser.setup()
    return parser

def find_chunk_by_criteria(chunks: List[CodeChunk], **criteria) -> CodeChunk:
    """Helper function to find a chunk matching given criteria"""
    for chunk in chunks:
        if all(getattr(chunk, k) == v for k, v in criteria.items()):
            return chunk
    return None

class TestCppParser:
    """Test suite for CppParser"""

    def test_parser_initialization(self, cpp_parser: CppParser):
        """Test parser initialization"""
        assert cpp_parser.parser is not None
        assert cpp_parser.language is not None
        assert cpp_parser.supports_extension(".cpp")
        assert cpp_parser.supports_extension(".hpp")
        assert cpp_parser.supports_extension(".h")

    def test_simple_file_parsing(self, cpp_parser: CppParser, test_config: AppConfig):
        """Test parsing a simple C++ file with function and class"""
        simple_cpp = """
        /**
         * A simple function that adds two numbers
         */
        int add(int a, int b) {
            return a + b;
        }

        /**
         * A simple class for demonstration
         */
        class SimpleClass {
        public:
            void hello() {
                std::cout << "Hello" << std::endl;
            }
        };
        """
        test_file = Path(test_config.db.codebase_directory) / "simple.cpp"
        test_file.write_text(simple_cpp)
        chunks = cpp_parser.parse_file(test_file)

        # Test function
        function_chunk = find_chunk_by_criteria(chunks, type=BaseChunkType.FUNCTION, name="add")
        assert function_chunk is not None
        assert "A simple function that adds two numbers" in function_chunk.docstring
        assert "int add(int a, int b)" in function_chunk.content

        # Test class
        class_chunk = find_chunk_by_criteria(chunks, type=BaseChunkType.CLASS, name="SimpleClass")
        assert class_chunk is not None
        assert "A simple class for demonstration" in class_chunk.docstring
        assert "class SimpleClass" in class_chunk.content

    def test_namespace_handling(self, cpp_parser: CppParser, test_config: AppConfig):
        """Test handling of namespaces"""
        namespace_cpp = """
        namespace test {
            /**
             * Function in namespace
             */
            void namespace_func() {}

            /**
             * Class in namespace
             */
            class NamespaceClass {
                void method() {}
            };
        }
        """
        test_file = Path(test_config.db.codebase_directory) / "namespace.cpp"
        test_file.write_text(namespace_cpp)
        chunks = cpp_parser.parse_file(test_file)

        func = find_chunk_by_criteria(chunks, type=BaseChunkType.FUNCTION, name="namespace_func")
        assert func is not None
        assert func.metadata.namespace == "test"

        cls = find_chunk_by_criteria(chunks, type=BaseChunkType.CLASS, name="NamespaceClass")
        assert cls is not None
        assert cls.metadata.namespace == "test"

    def test_template_handling(self, cpp_parser: CppParser, test_config: AppConfig):
        """Test handling of template classes and functions"""
        template_cpp = """
        /**
         * Template function
         */
        template<typename T>
        T max(T a, T b) {
            return a > b ? a : b;
        }

        /**
         * Template class
         */
        template<typename T, typename U>
        class Pair {
            T first;
            U second;
        };
        """
        test_file = Path(test_config.db.codebase_directory) / "template.cpp"
        test_file.write_text(template_cpp)
        chunks = cpp_parser.parse_file(test_file)

        func = find_chunk_by_criteria(chunks, type=BaseChunkType.FUNCTION, name="max")
        assert func is not None
        assert func.metadata.is_template
        assert "template<typename T>" in func.content

        cls = find_chunk_by_criteria(chunks, type=BaseChunkType.CLASS, name="Pair")
        assert cls is not None
        assert cls.metadata.is_template
        assert "template<typename T, typename U>" in cls.content

    def test_error_handling(self, cpp_parser: CppParser, test_config: AppConfig):
        """Test error handling for various error cases"""
        # Test invalid syntax
        invalid_file = Path(test_config.db.codebase_directory) / "invalid.cpp"
        invalid_file.write_text("class { invalid syntax")
        chunks = cpp_parser.parse_file(invalid_file)
        assert chunks == []

        # Test non-existent file
        nonexistent = Path(test_config.db.codebase_directory) / "nonexistent.cpp"
        chunks = cpp_parser.parse_file(nonexistent)
        assert chunks == []

        # Test non-C++ file
        non_cpp = Path(test_config.db.codebase_directory) / "readme.md"
        non_cpp.write_text("# README")
        chunks = cpp_parser.parse_file(non_cpp)
        assert chunks == []

    def test_preprocessor_handling(self, cpp_parser: CppParser, test_config: AppConfig):
        """Test handling of preprocessor directives"""
        preprocessor_cpp = """
        #include <iostream>
        #define MAX_SIZE 100

        /**
         * Class with preprocessor directives
         */
        class PreprocessorTest {
        #ifdef DEBUG
            void debug() {}
        #endif
            void release() {}
        };
        """
        test_file = Path(test_config.db.codebase_directory) / "preprocessor.cpp"
        test_file.write_text(preprocessor_cpp)
        chunks = cpp_parser.parse_file(test_file)

        cls = find_chunk_by_criteria(chunks, type=BaseChunkType.CLASS, name="PreprocessorTest")
        assert cls is not None
        assert "#ifdef DEBUG" in cls.content