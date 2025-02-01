# __future__ annotations is necessary for the type hints to work in this file
from __future__ import annotations
from dataclasses import dataclass
from typing import AsyncGenerator, List, Dict, Any, Optional
import chromadb
from pydantic import BaseModel
from pydantic_graph import BaseNode, EndStep, Graph, GraphRunContext, End, HistoryStep
from know_lang_bot.config import AppConfig
from know_lang_bot.utils.fancy_log import FancyLogger
from pydantic_ai import Agent
import logfire
from rich.pretty import Pretty
from enum import Enum
from rich.console import Console
from know_lang_bot.utils.model_provider import create_pydantic_model
from know_lang_bot.models.embeddings import generate_embedding

LOG = FancyLogger(__name__)
console = Console()

class ChatStatus(str, Enum):
    """Enum for tracking chat progress status"""
    STARTING = "starting"
    POLISHING = "polishing"
    RETRIEVING = "retrieving"
    ANSWERING = "answering"
    COMPLETE = "complete"
    ERROR = "error"

class StreamingChatResult(BaseModel):
    """Extended chat result with streaming information"""
    answer: str
    retrieved_context: Optional[RetrievedContext] = None
    status: ChatStatus
    progress_message: str
    
    @classmethod
    def from_node(cls, node: BaseNode, state: ChatGraphState) -> StreamingChatResult:
        """Create a StreamingChatResult from a node's current state"""
        if isinstance(node, PolishQuestionNode):
            return cls(
                answer="",
                status=ChatStatus.POLISHING,
                progress_message=f"Refining question: '{state.original_question}'"
            )
        elif isinstance(node, RetrieveContextNode):
            return cls(
                answer="",
                status=ChatStatus.RETRIEVING,
                progress_message=f"Searching codebase with: '{state.polished_question or state.original_question}'"
            )
        elif isinstance(node, AnswerQuestionNode):
            context_msg = f"Found {len(state.retrieved_context.chunks)} relevant segments" if state.retrieved_context else "No context found"
            return cls(
                answer="",
                retrieved_context=state.retrieved_context,
                status=ChatStatus.ANSWERING,
                progress_message=f"Generating answer... {context_msg}"
            )
        else:
            return cls(
                answer="",
                status=ChatStatus.ERROR,
                progress_message=f"Unknown node type: {type(node).__name__}"
            )
    
    @classmethod
    def complete(cls, result: ChatResult) -> StreamingChatResult:
        """Create a completed StreamingChatResult"""
        return cls(
            answer=result.answer,
            retrieved_context=result.retrieved_context,
            status=ChatStatus.COMPLETE,
            progress_message="Response complete"
        )
    
    @classmethod
    def error(cls, error_msg: str) -> StreamingChatResult:
        """Create an error StreamingChatResult"""
        return cls(
            answer=f"Error: {error_msg}",
            status=ChatStatus.ERROR,
            progress_message=f"An error occurred: {error_msg}"
        )

class RetrievedContext(BaseModel):
    """Structure for retrieved context"""
    chunks: List[str]
    metadatas: List[Dict[str, Any]]

class ChatResult(BaseModel):
    """Final result from the chat graph"""
    answer: str
    retrieved_context: Optional[RetrievedContext] = None

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
    config: AppConfig


# Graph Nodes
@dataclass
class PolishQuestionNode(BaseNode[ChatGraphState, ChatGraphDeps, ChatResult]):
    """Node that polishes the user's question"""
    system_prompt = """You are a code question refinement expert. Your ONLY task is to rephrase questions 
to be more precise for code context retrieval. Follow these rules strictly:

1. Output ONLY the refined question - no explanations or analysis
2. Preserve the original intent completely
3. Add missing technical terms if obvious
4. Keep the question concise - ideally one sentence
5. Focus on searchable technical terms
6. Do not add speculative terms not implied by the original question

Example Input: "How do I use transformers for translation?"
Example Output: "How do I use the Transformers pipeline for machine translation tasks?"

Example Input: "Where is the config stored?"
Example Output: "Where is the configuration file or configuration settings stored in this codebase?"
    """

    async def run(self, ctx: GraphRunContext[ChatGraphState, ChatGraphDeps]) -> RetrieveContextNode:
        # Create an agent for question polishing
        polish_agent = Agent(
            create_pydantic_model(
                model_provider=ctx.deps.config.llm.model_provider,
                model_name=ctx.deps.config.llm.model_name
            ),
            system_prompt=self.system_prompt
        )
        prompt = f"""Original question: "{ctx.state.original_question}"

Return ONLY the polished question - no explanations or analysis.
Focus on making the question more searchable while preserving its original intent."""
        
        result = await polish_agent.run(prompt)
        ctx.state.polished_question = result.data
        return RetrieveContextNode()

@dataclass
class RetrieveContextNode(BaseNode[ChatGraphState, ChatGraphDeps, ChatResult]):
    """Node that retrieves relevant code context"""
    
    async def run(self, ctx: GraphRunContext[ChatGraphState, ChatGraphDeps]) -> AnswerQuestionNode:
        try:
            question_embedding = generate_embedding(
                input=ctx.state.polished_question or ctx.state.original_question,
                config=ctx.deps.config.embedding
            )

            results = ctx.deps.collection.query(
                query_embeddings=question_embedding,
                n_results=ctx.deps.config.chat.max_context_chunks,
                include=['metadatas', 'documents', 'distances']
            )
            logfire.debug('query result: {result}', result=Pretty(results))
            
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

            ctx.state.retrieved_context = RetrievedContext(
                chunks=relevant_chunks,
                metadatas=relevant_metadatas,
            )
            
        except Exception as e:
            LOG.error(f"Error retrieving context: {e}")
        finally:
            return AnswerQuestionNode()

@dataclass
class AnswerQuestionNode(BaseNode[ChatGraphState, ChatGraphDeps, ChatResult]):
    """Node that generates the final answer"""
    system_prompt = """
You are an expert code assistant helping developers understand complex codebases. Follow these rules strictly:

1. ALWAYS START by directly answering the user's question - this is your primary task
2. Base your answer ONLY on the provided code context, not on general knowledge
3. When referencing code:
   - Cite specific files and line numbers
   - Quote relevant code snippets briefly
   - Explain why this code is relevant to the question
4. If you cannot find sufficient context to answer fully:
   - Clearly state what's missing
   - Explain what additional information would help
5. Focus on accuracy over comprehensiveness:
   - If you're unsure about part of your answer, explicitly say so
   - Better to acknowledge limitations than make assumptions

Remember: Your primary goal is answering the user's specific question, not explaining the entire codebase."""

    async def run(self, ctx: GraphRunContext[ChatGraphState, ChatGraphDeps]) -> End[ChatResult]:
        answer_agent = Agent(
            create_pydantic_model(
                model_provider=ctx.deps.config.llm.model_provider,
                model_name=ctx.deps.config.llm.model_name
            ),
            system_prompt=self.system_prompt
        )
        
        if not ctx.state.retrieved_context or not ctx.state.retrieved_context.chunks:
            return End(ChatResult(
                answer="I couldn't find any relevant code context for your question. "
                      "Could you please rephrase or be more specific?",
                retrieved_context=None,
            ))

        context = ctx.state.retrieved_context
        prompt = f"""
Question: {ctx.state.original_question}

Relevant Code Context:
{context.chunks}

Provide a focused answer to the question above. Structure your response as:
1. Direct Answer: Start with a clear, concise answer to the question
2. Supporting Evidence: Reference specific code with file locations
3. Limitations (if any): Note any missing context or uncertainties

Important: Stay focused on answering the specific question asked.
        """
        
        try:
            result = await answer_agent.run(prompt)
            return End(ChatResult(
                answer=result.data,
                retrieved_context=context,
            ))
        except Exception as e:
            LOG.error(f"Error generating answer: {e}")
            return End(ChatResult(
                answer="I encountered an error processing your question. Please try again.",
                retrieved_context=context,
            ))

# Create the graph
chat_graph = Graph(
    nodes=[PolishQuestionNode, RetrieveContextNode, AnswerQuestionNode]
)

async def process_chat(
    question: str,
    collection: chromadb.Collection,
    config: AppConfig
) -> ChatResult:
    """
    Process a chat question through the graph.
    This is the main entry point for chat processing.
    """
    state = ChatGraphState(original_question=question)
    deps = ChatGraphDeps(collection=collection, config=config)
    
    try:
        result, _history = await chat_graph.run(
            PolishQuestionNode(),
            state=state,
            deps=deps
        )
    except Exception as e:
        LOG.error(f"Error processing chat in graph: {e}")
        console.print_exception()
        
        result = ChatResult(
            answer="I encountered an error processing your question. Please try again."
        )
    finally:
        return result
    
async def stream_chat_progress(
    question: str,
    collection: chromadb.Collection,
    config: AppConfig
) -> AsyncGenerator[StreamingChatResult, None]:
    """
    Stream chat progress through the graph.
    This is the main entry point for chat processing.
    """
    state = ChatGraphState(original_question=question)
    deps = ChatGraphDeps(collection=collection, config=config)
    
    start_node = PolishQuestionNode()
    history: list[HistoryStep[ChatGraphState, ChatResult]] = []

    try:
        # Initial status
        yield StreamingChatResult(
            answer="",
            status=ChatStatus.STARTING,
            progress_message=f"Processing question: {question}"
        )

        with logfire.span(
            '{graph_name} run {start=}',
            graph_name='RAG_chat_graph',
            start=start_node,
        ) as run_span:
            current_node = start_node
            
            while True:
                # Yield current node's status before processing
                yield StreamingChatResult.from_node(current_node, state)
                
                try:
                    # Process the current node
                    next_node = await chat_graph.next(current_node, history, state=state, deps=deps, infer_name=False)
                    
                    if isinstance(next_node, End):
                        result: ChatResult = next_node.data
                        history.append(EndStep(result=next_node))
                        run_span.set_attribute('history', history)
                        # Yield final result
                        yield StreamingChatResult.complete(result)
                        return
                    elif isinstance(next_node, BaseNode):
                        current_node = next_node
                    else:
                        raise ValueError(f"Invalid node type: {type(next_node)}")
                        
                except Exception as node_error:
                    LOG.error(f"Error in node {current_node.__class__.__name__}: {node_error}")
                    yield StreamingChatResult.error(str(node_error))
                    return
                    
    except Exception as e:
        LOG.error(f"Error in stream_chat_progress: {e}")
        yield StreamingChatResult.error(str(e))
        return

    