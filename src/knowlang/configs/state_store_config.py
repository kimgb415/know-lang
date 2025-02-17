from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from knowlang.core.types import StateStoreProvider


class StateStoreConfig(BaseSettings):
    """Configuration for state storage"""
    type: StateStoreProvider = Field(
        default=StateStoreProvider.SQLITE,
        description="Type of state store to use"
    )
    store_path: Path = Field(
        default=Path("./file_state.db"),
        description="Path to store state data (for file-based stores)"
    )
    connection_url: Optional[str] = Field(
        default=None,
        description="Database connection URL (for network-based stores)"
    )
    pool_size: int = Field(
        default=5,
        description="Connection pool size for database connections"
    )
    max_overflow: int = Field(
        default=10,
        description="Maximum number of connections that can be created beyond pool_size"
    )
    pool_timeout: int = Field(
        default=30,
        description="Number of seconds to wait before timing out on getting a connection"
    )
    pool_recycle: int = Field(
        default=3600,
        description="Number of seconds after which to recycle connections"
    )
    echo: bool = Field(
        default=False,
        description="Echo SQL queries for debugging"
    )
    extra_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional store-specific configuration options"
    )

    @field_validator('store_path')
    def validate_store_path(cls, v: Path) -> Path:
        """Ensure store path parent directory exists"""
        if v.parent and not v.parent.exists():
            v.parent.mkdir(parents=True)
        return v

    def get_connection_args(self) -> Dict[str, Any]:
        """Get connection arguments based on store type"""
        if self.type == StateStoreProvider.SQLITE:
            return {
                'url': f'sqlite:///{self.store_path}',
                'pool_size': self.pool_size,
                'max_overflow': self.max_overflow,
                'pool_timeout': self.pool_timeout,
                'pool_recycle': self.pool_recycle,
                'echo': self.echo,
                **self.extra_config
            }
        else:
            raise ValueError(f"Unsupported state store type: {self.type}")