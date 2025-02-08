# __future__ annotations is necessary for the type hints to work in this file
from __future__ import annotations
from dataclasses import dataclass
from typing import AsyncGenerator, List, Dict, Any, Optional
import chromadb
from pydantic import BaseModel
from pydantic_graph import BaseNode, EndStep, Graph, GraphRunContext, End, HistoryStep
from know_lang_bot.configs.config import AppConfig, RerankerConfig, EmbeddingConfig
from know_lang_bot.utils.fancy_log import FancyLogger
from pydantic_ai import Agent
import logfire
from pprint import pformat
from enum import Enum
from rich.console import Console
from know_lang_bot.utils.model_provider import create_pydantic_model
from know_lang_bot.utils.chunking_util import truncate_chunk
from know_lang_bot.models.embeddings import EmbeddingInputType, generate_embedding
import voyageai
from voyageai.object.reranking import RerankingObject

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
    """Node that retrieves relevant code context using hybrid search: embeddings + reranking"""
    async def _get_initial_chunks(
        self, 
        query: str,
        embedding_config: EmbeddingConfig,
        collection: chromadb.Collection,
        n_results: int
    ) -> tuple[List[str], List[Dict], List[float]]:
        """Get initial chunks using embedding search"""
        question_embedding = generate_embedding(
            input=query,
            config=embedding_config,
            input_type=EmbeddingInputType.QUERY
        )
        
        results = collection.query(
            query_embeddings=question_embedding,
            n_results=n_results,
            include=['metadatas', 'documents', 'distances']
        )
        
        return (
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )

    async def _rerank_chunks(
        self,
        query: str,
        chunks: List[str],
        reranker_config: RerankerConfig,
    ) -> RerankingObject:
        """Rerank chunks using Voyage AI"""
        voyage_client = voyageai.Client()
        return voyage_client.rerank(
            query=query,
            documents=chunks,
            model=reranker_config.model_name,
            top_k=reranker_config.top_k,
            truncation=True
        )

    def _filter_by_distance(
        self,
        chunks: List[str],
        metadatas: List[Dict],
        distances: List[float],
        threshold: float
    ) -> tuple[List[str], List[Dict]]:
        """Filter chunks by distance threshold"""
        filtered_chunks = []
        filtered_metadatas = []
        
        for chunk, meta, dist in zip(chunks, metadatas, distances):
            if dist <= threshold:
                filtered_chunks.append(chunk)
                filtered_metadatas.append(meta)
                
        return filtered_chunks, filtered_metadatas
    
    async def run(self, ctx: GraphRunContext[ChatGraphState, ChatGraphDeps]) -> AnswerQuestionNode:
        try:
            # Get query
            query = ctx.state.polished_question or ctx.state.original_question
            
            # First pass: Get more candidates using embedding search
            initial_chunks, initial_metadatas, distances = await self._get_initial_chunks(
                query=query,
                embedding_config=ctx.deps.config.embedding,
                collection=ctx.deps.collection,
                n_results=min(ctx.deps.config.chat.max_context_chunks * 2, 50)
            )
                
            # Log top k initial results by distance
            top_k_initial = sorted(
                zip(initial_chunks, distances),
                key=lambda x: x[1]
            )[:ctx.deps.config.reranker.top_k]
            logfire.info('top k embedding search results: {results}', results=top_k_initial)
            top_k_initial_chunks = [chunk for chunk, _ in top_k_initial]
            
            # Only proceed to reranking if we have initial results
            if not initial_chunks:
                LOG.warning("No initial chunks found through embedding search")
                raise Exception("No chunks found through embedding search")

            # Second pass: Rerank the candidates
            try:
                if not ctx.deps.config.reranker.enabled:
                    raise Exception("Reranker is disabled")
                
                # Second pass: Rerank candidates
                reranking = await self._rerank_chunks(
                    query=query,
                    chunks=initial_chunks,
                    reranker_config=ctx.deps.config.reranker
                )
                logfire.info('top k reranking search results: {results}', results=reranking.results)
                
                # Build final context from reranked results
                relevant_chunks = []
                relevant_metadatas = []
                
                for result in reranking.results:
                    # Only include if score is good enough
                    if result.relevance_score >= ctx.deps.config.reranker.relevance_threshold:
                        relevant_chunks.append(result.document)
                        # Get corresponding metadata using original index
                        relevant_metadatas.append(initial_metadatas[result.index])
                    
                if not relevant_chunks:
                    raise Exception("No relevant chunks found through reranking")
                
                
            except Exception as e:
                # Fallback to distance-based filtering if reranking fails
                LOG.error(f"Reranking failed, falling back to distance-based filtering: {e}")
                relevant_chunks, relevant_metadatas = self._filter_by_distance(
                    chunks=top_k_initial_chunks,
                    metadatas=initial_metadatas,
                    distances=distances,
                    threshold=ctx.deps.config.chat.similarity_threshold
                )
            
            ctx.state.retrieved_context = RetrievedContext(
                chunks=relevant_chunks,
                metadatas=relevant_metadatas,
            )
            
        except Exception as e:
            LOG.error(f"Error in context retrieval: {e}")
            ctx.state.retrieved_context = RetrievedContext(chunks=[], metadatas=[])
        
        finally:
            return AnswerQuestionNode()

@dataclass
class AnswerQuestionNode(BaseNode[ChatGraphState, ChatGraphDeps, ChatResult]):
    """Node that generates the final answer"""
    system_prompt = """
You are an expert code assistant helping developers understand complex codebases. Follow these rules strictly:

1. ALWAYS answer the user's question - this is your primary task
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
        for chunk in context.chunks:
            chunk = truncate_chunk(chunk, ctx.deps.config.chat.max_length_per_chunk)

        prompt = f"""
Question: {ctx.state.original_question}

Relevant Code Context:
{context.chunks}

Provide a focused answer to the question based on the provided context.

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
            # Temporary fix to disable PolishQuestionNode
            RetrieveContextNode(),
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
    
    # Temporary fix to disable PolishQuestionNode
    start_node = RetrieveContextNode()
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

    