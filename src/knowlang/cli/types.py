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
    url: str
    output: Literal["table", "json"]
    command: Literal["parse"]  # for command identification

@dataclass
class ChatCommandArgs(BaseCommandArgs):
    """Arguments for the chat command."""
    command: Literal["chat"]
    port: Optional[int] = None
    share: bool = False
    server_port: Optional[int] = None
    server_name: Optional[str] = None

@dataclass
class ServeCommandArgs(BaseCommandArgs):
    """Arguments for the serve command."""
    command: Literal["serve"]
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False
    workers: int = 1