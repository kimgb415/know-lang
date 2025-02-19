from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from knowlang.configs.state_store_config import StateStoreConfig
from knowlang.core.types import StateStoreProvider
from knowlang.indexing.state_store.base import StateChangeType, StateStore
from knowlang.utils.fancy_log import FancyLogger

LOG = FancyLogger(__name__)

class PostgresStateStore(StateStore):
    """PostgreSQL implementation of state storage using SQLAlchemy"""

    def __init__(self, config: StateStoreConfig):
        """
        Initialize PostgreSQL database connection with configuration.
        Expects the configuration type to be StateStoreProvider.POSTGRES.
        """
        if config.type != StateStoreProvider.POSTGRES:
            raise ValueError(f"Invalid store type for PostgresStateStore: {config.type}")

        self.config = config
        connection_args = config.get_connection_args()  # Should return a dict including 'url'
        self.engine = create_engine(
            connection_args.pop('url'),
            **connection_args
        )
        self.Session = sessionmaker(bind=self.engine)
        LOG.debug(f"Initialized Postgres state store at {config.store_path}")