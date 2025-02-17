"""Command implementation for parsing codebases."""
from pathlib import Path
from typing import Optional

from knowlang.configs.config import AppConfig
from knowlang.parser.factory import CodeParserFactory
from knowlang.parser.providers.git import GitProvider
from knowlang.parser.providers.filesystem import FilesystemProvider
from knowlang.indexing.indexing_agent import IndexingAgent
from knowlang.cli.display.formatters import get_formatter
from knowlang.cli.display.progress import ProgressTracker
from knowlang.utils.fancy_log import FancyLogger
from knowlang.cli.types import ParseCommandArgs

LOG = FancyLogger(__name__)

def create_config(config_path: Optional[Path] = None) -> AppConfig:
    """Create configuration from file or defaults."""
    if config_path:
        with open(config_path, 'r') as file:
            config_data = file.read()
            return AppConfig.model_validate_json(config_data)
    return AppConfig()

async def parse_command(args: ParseCommandArgs) -> None:
    """Execute the parse command.
    
    Args:
        args: Typed command line arguments
    """
    # Load configuration
    config = create_config(args.config)
    
    # Update codebase directory in config
    config.db.codebase_directory = str(args.path)
    
    # Create parser factory
    factory = CodeParserFactory(config)
    
    # Determine provider
    source_path = args.path
    if (source_path / '.git').exists():
        LOG.info(f"Detected Git repository at {source_path}")
        provider = GitProvider(source_path, config)
    else:
        LOG.info(f"Using filesystem provider for {source_path}")
        provider = FilesystemProvider(source_path, config)
    
    # Process files
    total_chunks = []
    progress = ProgressTracker("Parsing files...")
    
    with progress.progress():
        for file_path in provider.get_files():
            progress.update(f"processing {file_path}...")
            
            parser = factory.get_parser(file_path)
            if parser:
                chunks = parser.parse_file(file_path)
                total_chunks.extend(chunks)
    
    # Display results
    if total_chunks:
        LOG.info(f"\nFound {len(total_chunks)} code chunks")
        formatter = get_formatter(args.output)
        formatter.display_chunks(total_chunks)
    else:
        LOG.warning("No code chunks found")
    
    # Process summaries
    summarizer = IndexingAgent(config)
    await summarizer.process_chunks(total_chunks)