from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings
from .base import generate_model_config


class LoggingConfig(BaseSettings):
    """Logging configuration settings"""
    model_config = generate_model_config(
        env_file=".env.logging",
    )
    level: str = Field(
        default="INFO",
        description="Default logging level"
    )
    json_logging: bool = Field(
        default=False, 
        description="Whether to use JSON format for logs"
    )
    file_enabled: bool = Field(
        default=False,
        description="Whether to enable file logging"
    )
    file_path: Path = Field(
        default=Path("logs/knowlang.log"),
        description="Path to log file"
    )
    show_path: bool = Field(
        default=True, 
        description="Show file path in Rich logs"
    )
    rich_tracebacks: bool = Field(
        default=True,
        description="Use Rich for exception tracebacks"
    )
    tracebacks_show_locals: bool = Field(
        default=True,
        description="Show local variables in Rich tracebacks"
    )

