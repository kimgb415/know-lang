"""Type definitions for CLI arguments."""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

@dataclass
class BaseCommandArgs:
    """Base arguments for all commands."""
    verbose: bool
    config: Optional[Path]

@dataclass
class ParseCommandArgs(BaseCommandArgs):
    """Arguments for the parse command."""
    path: Path
    output: Literal["table", "json"]
    command: Literal["parse"]  # for command identification