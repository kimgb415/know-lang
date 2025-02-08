"""Progress tracking for CLI operations."""
from contextlib import contextmanager
from typing import Iterator
from rich.console import Console

console = Console()

class ProgressTracker:
    """Track progress of file processing."""
    
    def __init__(self, description: str = "Processing..."):
        self.description = description
        self.status = None
        self.processed_files = 0
    
    @contextmanager
    def progress(self) -> Iterator[None]:
        """Context manager for tracking progress."""
        with console.status(f"[bold green]{self.description}") as status:
            self.status = status
            yield
            self.status = None
    
    def update(self, message: str) -> None:
        """Update progress status.
        
        Args:
            message: Current status message
        """
        if self.status:
            self.processed_files += 1
            self.status.update(
                f"[bold yellow]Processed {self.processed_files} files, "
                f"[bold green]{message}"
            )