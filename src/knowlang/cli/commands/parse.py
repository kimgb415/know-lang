"""Command implementation for parsing codebases."""
from pathlib import Path
from typing import Optional

from knowlang.configs.config import AppConfig
from knowlang.indexing.codebase_manager import CodebaseManager
from knowlang.indexing.increment_update import IncrementalUpdater
from knowlang.indexing.state_manager import StateManager
from knowlang.indexing.state_store.base import StateChangeType
from knowlang.parser.factory import CodeParserFactory
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
    config.db.codebase_directory = Path(args.path)
    
    # Create parser code_parser_factory
    code_parser_factory = CodeParserFactory(config)
    codebase_manager = CodebaseManager(config)
    state_manager = StateManager(config)
    
    # Process files
    total_chunks = []
    progress = ProgressTracker("Parsing Codebase...")
    
    with progress.progress():
        codebase_files = await codebase_manager.get_current_files()
        progress.update(f"detected {len(codebase_files)} files in codebase")
        file_changes = await state_manager.state_store.detect_changes(codebase_files)
        progress.update(f"detected {len(file_changes)} file changes")

        for changed_file_path in [
            (config.db.codebase_directory / change.path) 
            for change in file_changes
            if change.change_type != StateChangeType.DELETED
        ]:
            progress.update(f"parsing code in {changed_file_path}...")
            
            parser = code_parser_factory.get_parser(changed_file_path)
            if parser:
                chunks = parser.parse_file(changed_file_path)
                total_chunks.extend(chunks)
    
        updater = IncrementalUpdater(config)
        await updater.update_codebase(
            chunks=total_chunks, 
            file_changes=file_changes
        )

    # Display results
    if total_chunks:
        LOG.info(f"\nFound {len(total_chunks)} code chunks")
        formatter = get_formatter(args.output)
        formatter.display_chunks(total_chunks)
    else:
        LOG.warning("No code chunks found")
    
    # Process summaries