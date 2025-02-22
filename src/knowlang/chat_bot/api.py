from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from knowlang.configs import AppConfig
from knowlang.vector_stores import VectorStoreFactory
from knowlang.utils import FancyLogger
from knowlang.chat_bot import (
    stream_chat_progress, 
    ChatStatus,
    ChatAnalytics,
    StreamingChatResult,
)

LOG = FancyLogger(__name__)


class ServerSentChatEvent(BaseModel):
    event: ChatStatus
    data: StreamingChatResult

# Create FastAPI app
app = FastAPI(title="KnowLang API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your frontend origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = AppConfig()
# Dependency to get config
async def get_app_config():
    return config

# Dependency to get vector store
async def get_vector_store(config: AppConfig = Depends(get_app_config)):
    return VectorStoreFactory.get(config.db)

# Dependency to get chat analytics
async def get_chat_analytics(config: AppConfig = Depends(get_app_config)):
    return ChatAnalytics(config.chat_analytics)

@app.get("/api/v1/chat/stream")
async def stream_chat(
    query: str,
    config: AppConfig = Depends(get_app_config),
    vector_store = Depends(get_vector_store),
    chat_analytics = Depends(get_chat_analytics),
):
    """
    Streaming chat endpoint that uses server-sent events (SSE)
    """
    async def event_generator():
        # Process using the core logic from Gradio
        async for result in stream_chat_progress(query, vector_store, config):
            yield ServerSentChatEvent(event=result.status, data=result).model_dump()
                
    return EventSourceResponse(event_generator())