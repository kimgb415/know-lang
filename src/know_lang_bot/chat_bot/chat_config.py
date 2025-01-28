from pydantic_settings import BaseSettings
from pydantic import Field
from know_lang_bot.config import AppConfig

class ChatConfig(BaseSettings):
    max_context_chunks: int = Field(
        default=5,
        description="Maximum number of similar chunks to include in context"
    )
    similarity_threshold: float = Field(
        default=0.7,
        description="Minimum similarity score to include a chunk"
    )
    interface_title: str = Field(
        default="Code Repository Q&A Assistant",
        description="Title shown in the chat interface"
    )
    interface_description: str = Field(
        default="Ask questions about the codebase and I'll help you understand it!",
        description="Description shown in the chat interface"
    )

class ChatAppConfig(AppConfig):
    chat: ChatConfig = Field(default_factory=ChatConfig)


chat_app_config = ChatAppConfig()