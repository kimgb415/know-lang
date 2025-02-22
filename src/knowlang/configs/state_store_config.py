from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from knowlang.core.types import StateStoreProvider


class StateStoreConfig(BaseSettings):
    """Configuration for state storage"""
    provider: StateStoreProvider = Field(
        default=StateStoreProvider.SQLITE,
        description="state store provider to use"
    )
    store_path: Path = Field(
        default=Path("./statedb/file_state.db"),
        description="Path to store state data (for file-based stores)"
    )
    connection_url: Optional[str] = Field(
        default=None,
        description="Database connection URL (for network-based stores like PostgreSQL)"
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
        common_args = {
            'pool_size': self.pool_size,
            'max_overflow': self.max_overflow,
            'pool_timeout': self.pool_timeout,
            'pool_recycle': self.pool_recycle,
            'echo': self.echo,
            **self.extra_config,
        }
        
        if self.provider == StateStoreProvider.SQLITE:
            return {
                'url': f'sqlite:///{self.store_path}',
                **common_args,
            }
        elif self.provider == StateStoreProvider.POSTGRES:
            if not self.connection_url:
                raise ValueError("For PostgreSQL, the 'connection_url' must be provided.")
            return {
                'url': self.connection_url,
                **common_args,
            }