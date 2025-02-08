"""Argument parsing for KnowLang CLI."""
import argparse
from pathlib import Path
from typing import Union

from knowlang.cli.commands.chat import chat_command
from knowlang.cli.types import ChatCommandArgs, ParseCommandArgs, BaseCommandArgs
from knowlang.cli.commands.parse import parse_command

def _convert_to_args(parsed_args: argparse.Namespace) -> Union[ParseCommandArgs, BaseCommandArgs]:
    """Convert parsed namespace to typed arguments."""
    base_args = {
        "verbose": parsed_args.verbose,
        "config": parsed_args.config if hasattr(parsed_args, "config") else None
    }
    
    if parsed_args.command == "parse":
        return ParseCommandArgs(
            **base_args,
            path=parsed_args.path,
            output=parsed_args.output,
            command="parse"
        )
    elif parsed_args.command == "chat":
        return ChatCommandArgs(
            **base_args,
            command="chat",
            port=parsed_args.port,
            share=parsed_args.share,
            server_port=parsed_args.server_port,
            server_name=parsed_args.server_name
        )
    
    return BaseCommandArgs(**base_args)

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
    parse_parser.set_defaults(func=parse_command)

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
    chat_parser.set_defaults(func=chat_command)
    
    return parser

def parse_args() -> Union[ParseCommandArgs, BaseCommandArgs]:
    """Parse command line arguments into typed objects."""
    parser = create_parser()
    return _convert_to_args(parser.parse_args())