import os
from pathlib import Path
from typing import Generator

from git import Repo

from knowlang.configs import AppConfig
from knowlang.parser.base.provider import CodeProvider


class GitProvider(CodeProvider):
    """Provides code files from a Git repository"""
    def __init__(self, repo_path: Path, config: AppConfig):
        self.repo_path = repo_path
        self.config = config
        self._validate_repo()

    def _validate_repo(self):
        """Validate that the repository exists and is not bare"""
        if not (self.repo_path / '.git').exists():
            raise ValueError(f"No git repository found at {self.repo_path}")
        repo = Repo(self.repo_path)
        if repo.bare:
            raise ValueError(f"Repository {self.repo_path} is bare")

    def get_files(self) -> Generator[Path, None, None]:
        repo = Repo(self.repo_path)
        for dirpath, _, filenames in os.walk(repo.working_tree_dir):
            dir_path = Path(dirpath)
            if repo.ignored(dir_path) or not self.config.parser.path_patterns.should_process_path(dir_path):
                continue

            for filename in filenames:
                file_path = dir_path / filename
                if (not repo.ignored(file_path) and 
                    self.config.parser.path_patterns.should_process_path(file_path)):
                    yield file_path