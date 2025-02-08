from abc import ABC, abstractmethod
from typing import List
from pathlib import Path
from know_lang_bot.core.types import CodeChunk
from know_lang_bot.configs.config import AppConfig, LanguageConfig
from tree_sitter import Language, Parser

class LanguageParser(ABC):
    """Abstract base class for language-specific parsers"""
    
    def __init__(self, config: AppConfig):
        self.config : AppConfig = config
        self.language : Language = None
        self.parser : Parser = None
        self.language_config : LanguageConfig = None
    
    @abstractmethod
    def setup(self) -> None:
        """Set up the parser (e.g., initialize tree-sitter)"""
        pass

    @abstractmethod
    def parse_file(self, file_path: Path) -> List[CodeChunk]:
        """Parse a single file and return code chunks"""
        pass

    @abstractmethod
    def supports_extension(self, ext: str) -> bool:
        """Check if this parser supports a given file extension"""
        pass