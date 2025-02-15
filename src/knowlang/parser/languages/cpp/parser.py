# languages/cpp/parser.py
from typing import List, Optional
from pathlib import Path
import tree_sitter_cpp
from tree_sitter import Language, Parser, Node

from knowlang.parser.base.parser import LanguageParser
from knowlang.core.types import CodeChunk, BaseChunkType, CodeLocation, CodeMetadata, LanguageEnum
from knowlang.utils.fancy_log import FancyLogger

LOG = FancyLogger(__name__)

class CppParser(LanguageParser):
    """C++-specific implementation of LanguageParser"""
    
    def setup(self) -> None:
        """Initialize tree-sitter with C++ language support"""
        self.language_name = LanguageEnum.CPP
        self.language = Language(tree_sitter_cpp.language())
        self.parser = Parser(self.language)
        self.language_config = self.config.parser.languages["cpp"]
    
    def _get_preceding_docstring(self, node: Node, source_code: bytes) -> Optional[str]:
        """Extract docstring from comments"""
        docstring_parts = []
        current_node = node.prev_sibling

        while current_node:
            if current_node.type == "comment":
                comment = source_code[current_node.start_byte:current_node.end_byte].decode('utf-8')
                # Handle both // and /* */ style comments
                comment = comment.lstrip('/')
                comment = comment.strip('*')
                comment = comment.strip()
                docstring_parts.insert(0, comment)
            elif current_node.type not in ("empty_statement", "preproc_include", "declaration"):
                break
            current_node = current_node.prev_sibling
        
        return '\n'.join(docstring_parts) if docstring_parts else None

    def _get_namespace_context(self, node: Node, source_code: bytes) -> Optional[str]:
        """Get the namespace context of a node"""
        current = node.parent
        namespaces = []
        
        while current:
            if current.type == "namespace_definition":
                for child in current.children:
                    if child.type == "namespace_identifier":
                        namespaces.insert(0, source_code[child.start_byte:child.end_byte].decode('utf-8'))
                        break
            current = current.parent
            
        return "::".join(namespaces) if namespaces else None

    def _is_parent_template(self, node: Node) -> bool:
        """Check if a node is a template"""
        return node.parent and node.parent.type == "template_declaration"

    def _get_content(self, node: Node, source_code: bytes) -> str:
        """Get content of a node"""
        if self._is_parent_template(node):
            return source_code[node.parent.start_byte:node.parent.end_byte].decode('utf-8')
        else:
            return source_code[node.start_byte:node.end_byte].decode('utf-8')
    
    def _process_class(self, node: Node, source_code: bytes, file_path: Path) -> CodeChunk:
        """Process a class node and return a CodeChunk"""
        # In C++, we need to look for the identifier within a sequence of nodes
        name = node.child_by_field_name("name").text.decode('utf-8')

        if not name:
            raise ValueError(f"Could not find class name in node: {node.text}")
        
        return CodeChunk(
            language=self.language_name,
            type=BaseChunkType.CLASS,
            name=name,
            content=self._get_content(node, source_code),
            location=CodeLocation(
                start_line=node.start_point[0],
                end_line=node.end_point[0],
                file_path=str(file_path)
            ),
            docstring=self._get_preceding_docstring(node, source_code),
            metadata=CodeMetadata(
                namespace=self._get_namespace_context(node, source_code),
                is_template=self._is_parent_template(node)
            )
        )

    def _process_function(self, node: Node, source_code: bytes, file_path: Path) -> CodeChunk:
        """Process a function node and return a CodeChunk"""
        # Find function name - need to handle both normal functions and methods
        name = None
        for child in node.children:
            if child.type == "function_declarator":
                for subchild in child.children:
                    if subchild.type == "identifier" or "qualified_identifier":
                        name = subchild.text.decode('utf-8')
                        break
            if name:
                break
        
        if not name:
            raise ValueError(f"Could not find function name in node: {node.text}")
        

        return CodeChunk(
            type=BaseChunkType.FUNCTION,
            language=self.language_name,
            name=name,
            content=self._get_content(node, source_code),
            location=CodeLocation(
                start_line=node.start_point[0],
                end_line=node.end_point[0],
                file_path=str(file_path)
            ),
            docstring=self._get_preceding_docstring(node, source_code),
            metadata=CodeMetadata(
                namespace=self._get_namespace_context(node, source_code),
                is_template=self._is_parent_template(node)
            )
        )

    def parse_file(self, file_path: Path) -> List[CodeChunk]:
        """Parse a single C++ file and return list of code chunks"""
        if not self.supports_extension(file_path.suffix):
            LOG.debug(f"Skipping file {file_path}: unsupported extension")
            return []

        try:
            # Check file size limit
            if file_path.stat().st_size > self.language_config.max_file_size:
                LOG.warning(f"Skipping file {file_path}: exceeds size limit")
                return []

            with open(file_path, 'rb') as f:
                source_code = f.read()
            
            if not self.parser:
                raise RuntimeError("Parser not initialized. Call setup() first.")
                
            tree = self.parser.parse(source_code)
            chunks: List[CodeChunk] = []

            relative_path = file_path.relative_to(self.config.db.codebase_directory).as_posix()
            
            def traverse_node(node: Node):
                """Recursively traverse the syntax tree"""
                if node.type in ("class_specifier", "struct_specifier"):
                    try:
                        chunks.append(self._process_class(node, source_code, relative_path))
                    except ValueError as e:
                        LOG.warning(f"Failed to process class: {str(e)}")
                    finally:
                        # Don't process children of classes
                        return
                elif node.type == "function_definition":
                    try:
                        chunks.append(self._process_function(node, source_code, relative_path))
                    except ValueError as e:
                        LOG.warning(f"Failed to process function: {str(e)}")
                    finally:
                        # Don't process children of functions
                        return
                
                # Recursively process children since cpp has nested structures
                for child in node.children:
                    traverse_node(child)
            
            # Start traversal from root
            traverse_node(tree.root_node)
            return chunks

        except Exception as e:
            LOG.error(f"Error parsing file {file_path}: {str(e)}")
            return []
    
    def supports_extension(self, ext: str) -> bool:
        """Check if this parser supports a given file extension"""
        return ext.lower() in self.language_config.file_extensions