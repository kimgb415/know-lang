from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import chromadb
from pydantic import BaseModel
from pydantic_graph import BaseNode, Graph, GraphRunContext, End
import ollama
import logfire
from know_lang_bot.chat_bot.chat_config import ChatAppConfig, chat_app_config
from know_lang_bot.utils.fancy_log import FancyLogger
from pydantic_ai import Agent

LOG = FancyLogger(__name__)

# Data Models
class RetrievedContext(BaseModel):
    """Structure for retrieved context"""
    chunks: List[str]
    metadatas: List[Dict[str, Any]]
    references_md: str

class ChatResult(BaseModel):
    """Final result from the chat graph"""
    answer: str
    references_md: Optional[str] = None




@dataclass
class ChatGraphState:
    """State maintained throughout the graph execution"""
    original_question: str
    polished_question: Optional[str] = None
    retrieved_context: Optional[RetrievedContext] = None

@dataclass
class ChatGraphDeps:
    """Dependencies required by the graph"""
    collection: chromadb.Collection
    config: ChatAppConfig


# Graph Nodes
@dataclass
class PolishQuestion(BaseNode[ChatGraphState, ChatGraphDeps, ChatResult]):
    """Node that polishes the user's question"""
    system_prompt = """
    You are an expert at understanding code-related questions and reformulating them
    for better context retrieval. Your task is to polish the user's question to make
    it more specific and searchable. Focus on technical terms and code concepts.
    """

    async def run(self, ctx: GraphRunContext[ChatGraphState]) -> RetrieveContext:
        # Create an agent for question polishing
        from pydantic_ai import Agent
        polish_agent = Agent(
            f"{ctx.deps.config.llm.model_provider}:{ctx.deps.config.llm.model_name}"
        )
        prompt = f"""
        Original question: {ctx.state.original_question}
        
        Please reformulate this question to be more specific and searchable,
        focusing on technical terms and code concepts. Keep the core meaning
        but make it more precise for code context retrieval.
        """
        
        result = await polish_agent.run(prompt)
        ctx.state.polished_question = result.data
        return RetrieveContext()

@dataclass
class RetrieveContext(BaseNode[ChatGraphState, ChatGraphDeps, ChatResult]):
    """Node that retrieves relevant code context"""
    
    async def run(self, ctx: GraphRunContext[ChatGraphState]) -> AnswerQuestion:
        try:
            embedded_question = ollama.embed(
                model=ctx.deps.config.llm.embedding_model,
                input=ctx.state.polished_question or ctx.state.original_question
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
                file_path = meta['file_path'].split('/')[-1]
                ref = f"**{file_path}** (lines {meta['start_line']}-{meta['end_line']})"
                if meta.get('name'):
                    ref += f"\n- {meta['type']}: `{meta['name']}`"
                references.append(ref)
            
            ctx.state.retrieved_context = RetrievedContext(
                chunks=relevant_chunks,
                metadatas=relevant_metadatas,
                references_md="\n\n".join(references)
            )
            
            return AnswerQuestion()
            
        except Exception as e:
            LOG.error(f"Error retrieving context: {e}")
            return AnswerQuestion()

@dataclass
class AnswerQuestion(BaseNode[ChatGraphState, ChatGraphDeps, ChatResult]):
    """Node that generates the final answer"""
    system_prompt = """
    You are an expert code assistant helping users understand a codebase.
    Always:
    1. Reference specific files and line numbers in your explanations
    2. Be direct and concise while being comprehensive
    3. If the context is insufficient, explain why
    4. If you're unsure about something, acknowledge it
    """

    async def run(self, ctx: GraphRunContext[ChatGraphState]) -> End[ChatResult]:
        
        answer_agent = Agent(
            f"{ctx.deps.config.llm.model_provider}:{ctx.deps.config.llm.model_name}",
            system_prompt=self.system_prompt
        )
        
        if not ctx.state.retrieved_context or not ctx.state.retrieved_context.chunks:
            return End(ChatResult(
                answer="I couldn't find any relevant code context for your question. "
                      "Could you please rephrase or be more specific?",
                references_md=""
            ))

        context = ctx.state.retrieved_context
        prompt = f"""
        Question: {ctx.state.original_question}
        
        Available Code Context:
        {context.chunks}
        
        Please provide a comprehensive answer based on the code context above.
        Make sure to reference specific files and line numbers from the context.
        """
        
        try:
            result = await answer_agent.run(prompt)
            return End(ChatResult(
                answer=result.data,
                references_md=context.references_md
            ))
        except Exception as e:
            LOG.error(f"Error generating answer: {e}")
            return End(ChatResult(
                answer="I encountered an error processing your question. Please try again.",
                references_md=""
            ))

# Create the graph
chat_graph = Graph(
    nodes=[PolishQuestion, RetrieveContext, AnswerQuestion]
)

async def process_chat(
    question: str,
    collection: chromadb.Collection,
    config: ChatAppConfig
) -> ChatResult:
    """
    Process a chat question through the graph.
    This is the main entry point for chat processing.
    """
    state = ChatGraphState(original_question=question)
    deps = ChatGraphDeps(collection=collection, config=config)
    
    result, _history = await chat_graph.run(
        PolishQuestion(),
        state=state,
        deps=deps
    )
    
    return result