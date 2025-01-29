import argparse
import sys
from pathlib import Path
from typing import List, Optional
from rich.console import Console
from rich.table import Table

from know_lang_bot.code_parser.parser import CodeChunk
from know_lang_bot.config import AppConfig
from know_lang_bot.parser.factory import CodeParserFactory
from know_lang_bot.parser.providers.git import GitProvider
from know_lang_bot.parser.providers.filesystem import FilesystemProvider
from know_lang_bot.utils.fancy_log import FancyLogger

LOG = FancyLogger(__name__)
console = Console()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Know Lang Bot - Code Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--source_path",
        type=str,
        default=".",
        help="Path to the source code (git repository or directory)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom configuration file",
        default=None
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)"
    )

    return parser.parse_args()

def create_config(config_path: Optional[str] = None) -> AppConfig:
    """Create configuration, optionally from a file"""
    if config_path:
        with open(config_path, 'r') as file:
            config_data = file.read()
            return AppConfig.model_validate_json(config_data)
    return AppConfig()

def display_results_table(chunks : List[CodeChunk]):
    """Display parsed chunks in a rich table"""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Type")
    table.add_column("Name")
    table.add_column("File")
    table.add_column("Lines")
    table.add_column("Parent")
    
    for chunk in chunks:
        table.add_row(
            chunk.type.value,
            chunk.name or "N/A",
            str(chunk.file_path),
            f"{chunk.start_line}-{chunk.end_line}",
            chunk.parent_name or "N/A"
        )
    
    console.print(table)

def display_results_json(chunks: List[CodeChunk]):
    """Display parsed chunks as JSON"""
    import json
    print(json.dumps([chunk.model_dump() for chunk in chunks], indent=2))

def main():
    args = parse_args()
    
    # Setup logging
    if args.verbose:
        LOG.setLevel("DEBUG")
    
    try:
        # Load configuration
        config = create_config(args.config)
        source_path = Path(args.source_path)
        
        # Create parser factory
        factory = CodeParserFactory(config)
        
        # Determine provider
        if (source_path / '.git').exists():
            LOG.info(f"Detected Git repository at {source_path}")
            provider = GitProvider(source_path, config)
        else:
            LOG.info(f"Using filesystem provider for {source_path}")
            provider = FilesystemProvider(source_path, config)
        
        # Process files
        total_chunks = []
        with console.status("[bold green]Parsing files...") as status:
            for file_path in provider.get_files():
                status.update(f"[bold green]Processing {file_path}...")
                
                parser = factory.get_parser(file_path)
                if parser:
                    chunks = parser.parse_file(file_path)
                    total_chunks.extend(chunks)
        
        # Display results
        if total_chunks:
            LOG.info(f"\nFound {len(total_chunks)} code chunks")
            if args.output == "table":
                display_results_table(total_chunks)
            else:
                display_results_json(total_chunks)
        else:
            LOG.warning("No code chunks found")
        
    except Exception as e:
        LOG.error(f"Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()