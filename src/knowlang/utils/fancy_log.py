import json
import logging
from rich.console import Console
from rich.logging import RichHandler
from knowlang.configs import LoggingConfig


class JsonFormatter(logging.Formatter):
    """Format logs as JSON for structured logging."""
    def format(self, record):
        log_record = {
            "time": self.formatTime(record),
            "name": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "exc_info") and record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_record)


class FancyLogger(logging.Logger):
    """
    Enhanced logger using Rich for beautiful console output with optional file logging.
    """
    FORMAT = "%(message)s"
    JSON_FORMAT = '{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'

    def __init__(self, name: str):
        """
        Initialize the logger with Rich formatting and optional file output.
        
        Args:
            name: The logger name
        """
        config = LoggingConfig()
            
        logging.Logger.__init__(self, name, config.level)
        
        # Configure console logging with Rich
        console = Console()
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=config.show_path,
            markup=True,
            rich_tracebacks=config.rich_tracebacks,
            tracebacks_show_locals=config.tracebacks_show_locals
        )
        rich_handler.setLevel(logging._nameToLevel[config.level])
        rich_handler.setFormatter(logging.Formatter(self.FORMAT))
        self.addHandler(rich_handler)
        
        # Configure file logging if enabled
        if config.file_enabled:
            self._setup_file_logging(config)
    
    def _setup_file_logging(self, config: LoggingConfig):
        """Set up file logging with proper directory creation."""
        log_path = config.file_path
        log_dir = log_path.parent
        
        # Create log directory if it doesn't exist
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
        
        # Choose formatter based on JSON_LOGGING flag
        file_formatter = JsonFormatter() if config.json_logging else logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Create and configure file handler
        file_handler = logging.FileHandler(config.file_path)
        file_handler.setLevel(logging._nameToLevel[config.level])
        file_handler.setFormatter(file_formatter)
        self.addHandler(file_handler)


def get_logger(name: str) -> FancyLogger:
    """
    Get a configured logger instance.
    
    Args:
        name: The logger name
    
    Returns:
        A configured FancyLogger instance
    """
    return logging.getLogger(name)


def setup_logger() -> None:
    """
    Configure the global logging settings.
    """
    config = LoggingConfig()
    
    # Set the default logger class
    logging.setLoggerClass(FancyLogger)
    
    # Configure root logger
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
    
    root_logger = get_logger("knowlang")
    root_logger.setLevel(logging._nameToLevel[config.level])
    
    return root_logger
