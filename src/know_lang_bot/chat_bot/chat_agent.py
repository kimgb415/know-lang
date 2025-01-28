from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import chromadb
from pathlib import Path
from pydantic_ai import Agent, RunContext
from know_lang_bot.chat_bot.chat_config import ChatAppConfig, chat_app_config
from know_lang_bot.utils.fancy_log import FancyLogger
from pydantic import BaseModel
import ollama
import logfire

LOG = FancyLogger(__name__)

@dataclass
class CodeQADeps:
    """Dependencies for the Code Q&A Agent"""
    collection: chromadb.Collection
    config: ChatAppConfig

class RetrievedContext(BaseModel):
    """Structure for retrieved context"""
    chunks: List[str]
    metadatas: List[Dict[str, Any]]
    references_md: str

class AgentResponse(BaseModel):
    """Structure for agent responses"""
    answer: str
    references_md: Optional[str] = None

# Initialize the agent with system prompt and dependencies
code_qa_agent = Agent(
    f'{chat_app_config.llm.model_provider}:{chat_app_config.llm.model_name}',
    deps_type=CodeQADeps,
    result_type=AgentResponse,
    system_prompt="""
    You are an expert code assistant helping users understand a codebase.
    
    Always:
    1. Reference specific files and line numbers in your explanations
    2. Be direct and concise while being comprehensive
    3. If the context is insufficient, explain why
    4. If you're unsure about something, acknowledge it

    Your response should be helpful for software engineers trying to understand complex codebases.
    """,
)

@code_qa_agent.tool
@logfire.instrument()
async def retrieve_context(
    ctx: RunContext[CodeQADeps], 
    question: str
) -> RetrievedContext:
    """
    Retrieve relevant code context from the vector database.
    
    Args:
        ctx: The context containing dependencies
        question: The user's question to find relevant code for
    """
    embedded_question = ollama.embed(
        model=ctx.deps.config.llm.embedding_model,
        input=question
    )

    results = ctx.deps.collection.query(
        query_embeddings=embedded_question['embeddings'],
        n_results=ctx.deps.config.chat.max_context_chunks,
        include=['metadatas', 'documents', 'distances']
    )
    
    relevant_chunks = []
    relevant_metadatas = []
    
    for doc, meta, dist in zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ):
        if dist <= ctx.deps.config.chat.similarity_threshold:
            relevant_chunks.append(doc)
            relevant_metadatas.append(meta)


    # Format references for display
    references = []
    for meta in relevant_metadatas:
        file_path = Path(meta['file_path']).name
        ref = f"**{file_path}** (lines {meta['start_line']}-{meta['end_line']})"
        if meta.get('name'):
            ref += f"\n- {meta['type']}: `{meta['name']}`"
        references.append(ref)
    
    return RetrievedContext(
        chunks=relevant_chunks,
        metadatas=relevant_metadatas,
        references_md="\n\n".join(references)
    )
    