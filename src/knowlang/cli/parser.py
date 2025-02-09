"""Argument parsing for KnowLang CLI."""
import argparse
from pathlib import Path
from typing import Optional, Sequence, Union, Dict, Type, Callable

from knowlang.cli.commands.chat import chat_command
from knowlang.cli.commands.parse import parse_command
from knowlang.cli.types import ChatCommandArgs, ParseCommandArgs, BaseCommandArgs

# Define command configurations
COMMAND_CONFIGS: Dict[str, tuple[Type[BaseCommandArgs], Callable]] = {
    "parse": (ParseCommandArgs, parse_command),
    "chat": (ChatCommandArgs, chat_command),
}

def _convert_to_args(parsed_namespace: argparse.Namespace) -> Union[ParseCommandArgs, ChatCommandArgs]:
    """Convert parsed namespace to typed arguments."""
    base_args = {
        "verbose": parsed_namespace.verbose,
        "config": parsed_namespace.config if hasattr(parsed_namespace, "config") else None,
        "command": parsed_namespace.command
    }
    
    # Get the appropriate argument class and command function
    args_class, command_func = COMMAND_CONFIGS[parsed_namespace.command]
    
    if parsed_namespace.command == "parse":
        args = args_class(
            **base_args,
            path=Path(parsed_namespace.path).resolve(),
            output=parsed_namespace.output
        )
    elif parsed_namespace.command == "chat":
        args = args_class(
            **base_args,
            port=parsed_namespace.port,
            share=parsed_namespace.share,
            server_port=parsed_namespace.server_port,
            server_name=parsed_namespace.server_name
        )
    else:
        raise ValueError(f"Unknown command: {parsed_namespace.command}")
        
    args.command_func = command_func
    return args

def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        description="KnowLang - Code Understanding Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Global options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to custom configuration file",
        default=None
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(
        title="commands",
        description="Available commands",
        dest="command"
    )
    subparsers.required = True
    
    # Parse command
    parse_parser = subparsers.add_parser(
        "parse",
        help="Parse and index a codebase"
    )
    parse_parser.add_argument(
        "--output",
        type=str,
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)"
    )
    parse_parser.add_argument(
        "path",
        type=Path,
        help="Path to codebase directory or repository URL"
    )

    # Chat command
    chat_parser = subparsers.add_parser(
        "chat",
        help="Launch the chat interface"
    )
    chat_parser.add_argument(
        "--port",
        type=int,
        help="Port to run the interface on"
    )
    chat_parser.add_argument(
        "--share",
        action="store_true",
        help="Create a shareable link"
    )
    chat_parser.add_argument(
        "--server-port",
        type=int,
        help="Port to run the server on (if different from --port)"
    )
    chat_parser.add_argument(
        "--server-name",
        type=str,
        help="Server name to listen on (default: 0.0.0.0)"
    )
    
    return parser

def parse_args(args: Optional[Sequence[str]] = None) -> Union[ParseCommandArgs, BaseCommandArgs]:
    """Parse command line arguments into typed objects."""
    parser = create_parser()
    parsed_namespace = parser.parse_args(args)
    return _convert_to_args(parsed_namespace)