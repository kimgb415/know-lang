"""Command implementation for the API server interface."""
import uvicorn
from knowlang.cli.types import ServeCommandArgs
from knowlang.configs import AppConfig
from knowlang.utils import FancyLogger
from knowlang.vector_stores.factory import VectorStoreFactory
from knowlang.vector_stores import VectorStoreError

LOG = FancyLogger(__name__)

def create_config(args: ServeCommandArgs) -> AppConfig:
    """Create configuration from file or defaults."""
    if args.config:
        with open(args.config, 'r') as file:
            config_data = file.read()
            return AppConfig.model_validate_json(config_data)
    return AppConfig()

async def serve_command(args: ServeCommandArgs) -> None:
    """Execute the serve command.
    
    Args:
        args: Typed command line arguments
    """
    config = create_config(args)
    
    # Initialize vector store
    try:
        VectorStoreFactory.get(config.db, config.embedding)
    except VectorStoreError as e:
        LOG.error(
            "Vector store initialization failed. Please run 'knowlang parse' first to index your codebase."
            f"\nError: {str(e)}"
        )
        return

    # Configure uvicorn server using Server class directly
    config = uvicorn.Config(
        "knowlang.chat_bot.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
    )
    
    server = uvicorn.Server(config)
    # Use await instead of run() to respect the existing event loop
    await server.serve()