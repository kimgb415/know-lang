from knowlang.configs.state_store_config import StateStoreConfig
from knowlang.core.types import StateStoreProvider
from knowlang.indexing.state_store.base import StateChangeType, StateStore
from knowlang.utils.fancy_log import FancyLogger

LOG = FancyLogger(__name__)

class SQLiteStateStore(StateStore):
    """SQLite implementation of state storage using SQLAlchemy"""
    
    def __init__(self, config: StateStoreConfig):
        """Initialize SQLite database with configuration"""
        if config.type != StateStoreProvider.SQLITE:
            raise ValueError(f"Invalid store type for SQLiteStateStore: {config.type}")
            
        self.config = config
        connection_args = config.get_connection_args()
        
        self.engine = create_engine(
            connection_args.pop('url'),
            **connection_args
        )
        self.Session = sessionmaker(bind=self.engine)
        LOG.debug(f"Initialized SQLite state store at {config.store_path}")