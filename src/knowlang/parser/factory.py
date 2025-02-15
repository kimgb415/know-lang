from typing import Dict, Type, Optional
from pathlib import Path

from knowlang.parser.base.parser import LanguageParser
from knowlang.parser.languages.cpp.parser import CppParser
from knowlang.parser.languages.python.parser import PythonParser
from knowlang.configs.config import AppConfig

class CodeParserFactory():
    """Concrete implementation of parser factory"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self._parsers: Dict[str, LanguageParser] = {}
        self._parser_classes = self._register_parsers()
    
    def _register_parsers(self) -> Dict[str, Type[LanguageParser]]:
        """Register available parser implementations"""
        return {
            "python": PythonParser,
            'cpp': CppParser,
            # Add more languages here
        }
    
    def get_parser(self, file_path: Path) -> Optional[LanguageParser]:
        """Get appropriate parser for a file"""
        extension = file_path.suffix
        
        # Find parser class for this extension
        for lang, parser_class in self._parser_classes.items():
            if not self.config.parser.languages.get(lang, None) or not self.config.parser.languages[lang].enabled:
                continue
                
            parser = self._parsers.get(lang)
            if parser is None:
                parser = parser_class(self.config)
                parser.setup()
                self._parsers[lang] = parser

            if parser.supports_extension(extension):
                return self._parsers[lang]
        
        return None