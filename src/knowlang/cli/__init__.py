"""CLI entry points for KnowLang."""
import asyncio
from typing import Optional, Sequence

from knowlang.cli.parser import parse_args
from knowlang.utils.fancy_log import FancyLogger

LOG = FancyLogger(__name__)

async def main(args: Optional[Sequence[str]] = None) -> int:
    """Main CLI entry point.
    
    Args:
        args: Command line arguments. If None, sys.argv[1:] is used.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parsed_args = parse_args()
    
    # Setup logging
    if parsed_args.verbose:
        LOG.setLevel("DEBUG")
    
    try:
        # Execute command
        await parsed_args.command_func(parsed_args)
        return 0
    except Exception as e:
        LOG.error(f"Error: {str(e)}")
        if parsed_args.verbose:
            import traceback
            traceback.print_exc()
        return 1

def cli_main() -> None:
    """Entry point for CLI scripts."""
    exit_code = asyncio.run(main())
    exit(exit_code)