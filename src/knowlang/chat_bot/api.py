from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from knowlang.configs import AppConfig
from knowlang.vector_stores.factory import VectorStoreFactory
from knowlang.utils import FancyLogger
from knowlang.chat_bot import (
    stream_chat_progress, 
    ChatStatus,
    ChatAnalytics,
    StreamingChatResult,
)
from knowlang.api import ApiModelRegistry

LOG = FancyLogger(__name__)

@ApiModelRegistry.register
class ServerSentChatEvent(BaseModel):
    event: ChatStatus
    data: StreamingChatResult

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="KnowLang API",
        version="1.0.0",
        description="API for KnowLang code understanding assistant",
        routes=app.routes,
    )
    
    # Add our models to components/schemas
    openapi_schema["components"]["schemas"].update(ApiModelRegistry.get_all_schemas())
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

# Create FastAPI app
app = FastAPI(title="KnowLang API")
app.openapi = custom_openapi

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
    return VectorStoreFactory.get(config.db, config.embedding)

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