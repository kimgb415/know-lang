import os
from typing import List, Optional, Dict
from enum import Enum
from pathlib import Path
from tree_sitter import Language, Parser, Node
import tree_sitter_python
from pydantic import BaseModel
from git import Repo
from know_lang_bot.utils.fancy_log import FancyLogger

LOG = FancyLogger(__name__)


class ChunkType(str, Enum):
    CLASS = "class"
    FUNCTION = "function"
    OTHER = "other"

class CodeChunk(BaseModel):
    """Represents a chunk of code with its metadata"""
    type: ChunkType
    content: str
    start_line: int
    end_line: int
    file_path: str
    name: Optional[str] = None
    parent_name: Optional[str] = None  # For nested classes/functions
    docstring: Optional[str] = None

class CodeParser:
    parser : Parser = None
    laguage: Language = None

    def __init__(self, repo_path: str):
        """Initialize the parser with a repository path"""
        self.repo_path = Path(repo_path)
        self._init_tree_sitter()

    def _init_tree_sitter(self):
        """Initialize tree-sitter with Python language support"""
        # In real implementation, we'd need to handle language loading more robustly
        # For MVP, we'll assume Python parser is available
        self.language = Language(tree_sitter_python.language())
        self.parser = Parser(self.language)

    def _get_preceding_docstring(self, node: Node, source_code: bytes) -> Optional[str]:
        """Extract docstring from comments"""
        docstring_parts = []
        current_node : Node = node.prev_sibling

        while current_node:
            print(current_node.text)
            if current_node.type == "comment":
                comment = source_code[current_node.start_byte:current_node.end_byte].decode('utf-8')
                docstring_parts.insert(0, comment)
            elif current_node.type == "expression_statement":
                string_node = current_node.children[0] if current_node.children else None
                if string_node and string_node.type in ("string", "string_literal"):
                    docstring = source_code[string_node.start_byte:string_node.end_byte].decode('utf-8')
                    docstring_parts.insert(0, docstring)
                    
                break
            elif current_node.type not in ("empty_statement", "newline"):
                break
            current_node = current_node.prev_sibling
        
        return '\n'.join(docstring_parts) if docstring_parts else None

    def _has_syntax_error(self, node: Node) -> bool:
        """Check if the node or its children contain syntax errors"""
        if node.type == "ERROR":
            return True
        if node.has_error:
            return True
        return any(self._has_syntax_error(child) for child in node.children)

    def parse_file(self, file_path: Path) -> List[CodeChunk]:
        """Parse a single file and return list of code chunks"""
        if not file_path.suffix == '.py':
            LOG.warning(f"Skipping non-Python file: {file_path}")
            return []

        try:
            with open(file_path, 'rb') as f:
                source_code = f.read()
            
            tree = self.parser.parse(source_code)

            # Check for overall syntax validity
            if self._has_syntax_error(tree.root_node):
                LOG.warning(f"Syntax errors found in {file_path}")
                return []

            chunks: List[CodeChunk] = []
            
            # Process the syntax tree
            for node in tree.root_node.children:
                if node.type == "class_definition":
                    chunks.append(self._process_class(node, source_code, file_path))
                elif node.type == "function_definition":
                    chunks.append(self._process_function(node, source_code, file_path))
                else:
                    # Skip other node types for now
                    pass
            
            return chunks
        except Exception as e:
            LOG.error(f"Error parsing file {file_path}: {str(e)}")
            return []

    def _process_class(self, node: Node, source_code: bytes, file_path: Path) -> CodeChunk:
        """Process a class node and return a CodeChunk"""
        name = next(child.text.decode('utf-8') 
                   for child in node.children 
                   if child.type == "identifier")
        
        if not name:
            raise ValueError(f"Could not find class name in node: {node.text}")
        
        return CodeChunk(
            type=ChunkType.CLASS,
            name=name,
            content=source_code[node.start_byte:node.end_byte].decode('utf-8'),
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            file_path=str(file_path),
            docstring=self._get_preceding_docstring(node, source_code)
        )

    def _process_function(self, node: Node, source_code: bytes, file_path: Path) -> CodeChunk:
        """Process a function node and return a CodeChunk"""
        name = next(child.text.decode('utf-8') 
                   for child in node.children 
                   if child.type == "identifier")

        if not name:
            raise ValueError(f"Could not find function name in node: {node.text}")
        
        return CodeChunk(
            type=ChunkType.FUNCTION,
            name=name,
            content=source_code[node.start_byte:node.end_byte].decode('utf-8'),
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            file_path=str(file_path),
            docstring=self._get_preceding_docstring(node, source_code)
        )

    def parse_repository(self) -> List[CodeChunk]:
        """Parse all Python files in the repository"""
        chunks: List[CodeChunk] = []

        try:
            repo = Repo(self.repo_path)

            if repo.bare:
                raise ValueError(f"Repository {self.repo_path} is bare and has no working directory")

            for dirpath, _, filenames in os.walk(repo.working_tree_dir):
                if repo.ignored(dirpath):
                    LOG.debug(f"Skipping ignored directory: {dirpath}")
                    continue

                for file in filenames:
                    file_path = Path(dirpath) / file

                    if repo.ignored(file_path):
                        LOG.debug(f"Skipping ignored file: {file_path}")
                        continue

                    if file.endswith('.py'):
                        chunks.extend(self.parse_file(file_path))
        except Exception as e:
            LOG.error(f"Error processing repository: {str(e)}")
        
        return chunks
