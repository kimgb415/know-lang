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
    MODULE = "module"
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

    def _extract_docstring(self, node: Node, source_code: bytes) -> Optional[str]:
        """Extract docstring from a class or function node"""
        for child in node.children:
            if child.type == "expression_statement":
                string_node = child.children[0]
                if string_node.type in ("string", "string_literal"):
                    return source_code[string_node.start_byte:string_node.end_byte].decode('utf-8')
        return None

    def parse_file(self, file_path: Path) -> List[CodeChunk]:
        """Parse a single file and return list of code chunks"""
        if not file_path.suffix == '.py':
            LOG.warning(f"Skipping non-Python file: {file_path}")
            return []

        try:
            with open(file_path, 'rb') as f:
                source_code = f.read()
            
            tree = self.parser.parse(source_code)
            chunks: List[CodeChunk] = []
            
            # Process the syntax tree
            for node in tree.root_node.children:
                if node.type == "class_definition":
                    chunks.append(self._process_class(node, source_code, file_path))
                elif node.type == "function_definition":
                    chunks.append(self._process_function(node, source_code, file_path))
                else:
                    # Store other top-level code as separate chunks
                    if node.type not in ("comment", "empty_statement"):
                        chunks.append(CodeChunk(
                            type=ChunkType.OTHER,
                            content=source_code[node.start_byte:node.end_byte].decode('utf-8'),
                            start_line=node.start_point[0],
                            end_line=node.end_point[0],
                            file_path=str(file_path)
                        ))
            
            return chunks
        except Exception as e:
            LOG.error(f"Error parsing file {file_path}: {str(e)}")
            return []

    def _process_class(self, node: Node, source_code: bytes, file_path: Path) -> CodeChunk:
        """Process a class node and return a CodeChunk"""
        name = next(child.text.decode('utf-8') 
                   for child in node.children 
                   if child.type == "identifier")
        
        return CodeChunk(
            type=ChunkType.CLASS,
            name=name,
            content=source_code[node.start_byte:node.end_byte].decode('utf-8'),
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            file_path=str(file_path),
            docstring=self._extract_docstring(node, source_code)
        )

    def _process_function(self, node: Node, source_code: bytes, file_path: Path) -> CodeChunk:
        """Process a function node and return a CodeChunk"""
        name = next(child.text.decode('utf-8') 
                   for child in node.children 
                   if child.type == "identifier")
        
        return CodeChunk(
            type=ChunkType.FUNCTION,
            name=name,
            content=source_code[node.start_byte:node.end_byte].decode('utf-8'),
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            file_path=str(file_path),
            docstring=self._extract_docstring(node, source_code)
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
