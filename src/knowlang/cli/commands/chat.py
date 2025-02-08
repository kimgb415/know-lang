"""Command implementation for the chat interface."""
import chromadb
from knowlang.chat_bot.chat_interface import create_chatbot
from knowlang.configs.config import AppConfig
from knowlang.cli.types import ChatCommandArgs
from knowlang.utils.fancy_log import FancyLogger

LOG = FancyLogger(__name__)

def create_config(args: ChatCommandArgs) -> AppConfig:
    """Create configuration from file or defaults."""
    if args.config:
        with open(args.config, 'r') as file:
            config_data = file.read()
            return AppConfig.model_validate_json(config_data)
    return AppConfig()

async def chat_command(args: ChatCommandArgs) -> None:
    """Execute the chat command.
    
    Args:
        args: Typed command line arguments
    """
    config = create_config(args)
    
    # Verify database exists
    try:
        db_client = chromadb.PersistentClient(path=str(config.db.persist_directory))
        db_client.get_collection(name=config.db.collection_name)
    except Exception as e:
        LOG.error(
            "Database not found. Please run 'knowlang parse' first to index your codebase."
            f"\nError: {str(e)}"
        )
        return
    
    # Create and launch chatbot
    demo = create_chatbot(config)
    
    launch_kwargs = {
        "server_port": args.server_port,
        "server_name": args.server_name or "127.0.0.1",
        "share": args.share,
    }
    if args.port:
        launch_kwargs["port"] = args.port
        
    demo.launch(**launch_kwargs)