import os
from pathlib import Path
from typing import Generator

from knowlang.configs import AppConfig
from knowlang.parser.base.provider import CodeProvider


class FilesystemProvider(CodeProvider):
    """Provides code files from a filesystem directory"""
    def __init__(self, directory: Path, config: AppConfig):
        self.directory = directory
        self.config = config

    def get_files(self) -> Generator[Path, None, None]:
        for dirpath, _, filenames in os.walk(self.directory):
            dir_path = Path(dirpath)
            if not self.config.parser.path_patterns.should_process_path(dir_path):
                continue

            for filename in filenames:
                file_path = dir_path / filename
                if self.config.parser.path_patterns.should_process_path(file_path):
                    yield file_path