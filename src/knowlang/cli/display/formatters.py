"""Output formatters for CLI results."""
from typing import List, Protocol
import json
from rich.console import Console
from rich.table import Table

from knowlang.core.types import CodeChunk

console = Console()

class OutputFormatter(Protocol):
    """Protocol for output formatters."""
    
    def display_chunks(self, chunks: List[CodeChunk]) -> None:
        """Display code chunks in the appropriate format."""
        ...

class TableFormatter:
    """Format output as a rich table."""
    
    def display_chunks(self, chunks: List[CodeChunk]) -> None:
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
                chunk.location.to_single_line(),
            )
        
        console.print(table)

class JsonFormatter:
    """Format output as JSON."""
    
    def display_chunks(self, chunks: List[CodeChunk]) -> None:
        print(json.dumps([chunk.model_dump() for chunk in chunks], indent=2))

def get_formatter(format_type: str) -> OutputFormatter:
    """Get the appropriate formatter for the specified format.
    
    Args:
        format_type: The type of formatter to use ("table" or "json")
    
    Returns:
        An OutputFormatter instance
    
    Raises:
        ValueError: If format_type is not recognized
    """
    formatters = {
        "table": TableFormatter,
        "json": JsonFormatter
    }
    
    if format_type not in formatters:
        raise ValueError(f"Unknown format type: {format_type}")
    
    return formatters[format_type]()