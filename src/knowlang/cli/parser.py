"""Argument parsing for KnowLang CLI."""
import argparse
from pathlib import Path
from typing import Union

from knowlang.cli.types import ParseCommandArgs, BaseCommandArgs
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
    
    return parser

def parse_args() -> Union[ParseCommandArgs, BaseCommandArgs]:
    """Parse command line arguments into typed objects."""
    parser = create_parser()
    return _convert_to_args(parser.parse_args())