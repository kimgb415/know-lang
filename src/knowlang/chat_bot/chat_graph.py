# __future__ annotations is necessary for the type hints to work in this file
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional
import logfire
import voyageai
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_graph import (
    BaseNode, 
    End, 
    EndStep, 
    Graph, 
    GraphRunContext,
    HistoryStep
)
from rich.console import Console
from voyageai.object.reranking import RerankingObject
from knowlang.configs import AppConfig, EmbeddingConfig, RerankerConfig
from knowlang.models import EmbeddingInputType, generate_embedding
from knowlang.utils import create_pydantic_model, truncate_chunk
import logging
from knowlang.vector_stores import SearchResult, VectorStore
from knowlang.api import ApiModelRegistry

LOG = logging.getLogger(__name__)
console = Console()

@ApiModelRegistry.register
class ChatStatus(str, Enum):
    """Enum for tracking chat progress status"""
    STARTING = "starting"
    POLISHING = "polishing"
    RETRIEVING = "retrieving"
    ANSWERING = "answering"
    COMPLETE = "complete"
    ERROR = "error"

@ApiModelRegistry.register
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

@ApiModelRegistry.register
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
    vector_store: VectorStore
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
    
    async def _get_initial_results(
        self,
        query: str,
        embedding_config: EmbeddingConfig,
        vector_store: VectorStore,
        n_results: int
    ) -> List[SearchResult]:
        """Get initial results using embedding search"""
        query_embedding = generate_embedding(
            input=query,
            config=embedding_config,
            input_type=EmbeddingInputType.QUERY
        )
        
        return await vector_store.search(
            query_embedding=query_embedding,
            top_k=n_results
        )

    async def _rerank_results(
        self,
        query: str,
        results: List[SearchResult],
        reranker_config: RerankerConfig,
    ) -> List[SearchResult]:
        """Rerank results using Voyage AI"""
        try:
            voyage_client = voyageai.Client()
            reranking : RerankingObject = await voyage_client.rerank(
                query=query,
                documents=[r.document for r in results],
                model=reranker_config.model_name,
                top_k=reranker_config.top_k,
                truncation=True
            )
            
            # Convert reranking results back to SearchResults
            reranked_results: List[SearchResult] = []
            for rerank_result in reranking.results:
                if rerank_result.relevance_score >= reranker_config.relevance_threshold:
                    # Get the original result to preserve metadata
                    original_result : SearchResult = results[rerank_result.index]
                    reranked_results.append(SearchResult(
                        document=rerank_result.document,
                        metadata=original_result.metadata,
                        score=rerank_result.relevance_score
                    ))
            
            return reranked_results
            
        except Exception as e:
            LOG.error(f"Reranking failed: {e}")
            raise
    
    async def run(self, ctx: GraphRunContext[ChatGraphState, ChatGraphDeps]) -> AnswerQuestionNode:
        try:
            # Get query
            query = ctx.state.polished_question or ctx.state.original_question
            
            # First pass: Get candidates using embedding search
            initial_results = await self._get_initial_results(
                query=query,
                embedding_config=ctx.deps.config.embedding,
                vector_store=ctx.deps.vector_store,
                n_results=min(ctx.deps.config.chat.max_context_chunks * 2, 50)
            )
            
            if not initial_results:
                LOG.warning("No initial results found through embedding search")
                raise Exception("No results found through embedding search")
                
            # Log initial results
            logfire.info('initial embedding search results: {results}', 
                results=[(r.document, r.score) for r in initial_results])

            # Filter initial results to top_k by score
            initial_results = sorted(
                initial_results, 
                key=lambda x: x.score, 
                reverse=True
            )[:ctx.deps.config.reranker.top_k]

            try:
                if not ctx.deps.config.reranker.enabled:
                    raise Exception("Reranker is disabled")
                
                # Second pass: Rerank candidates
                reranked_results = await self._rerank_results(
                    query=query,
                    results=initial_results,
                    reranker_config=ctx.deps.config.reranker
                )
                
                if not reranked_results:
                    raise Exception("No relevant results found through reranking")
                
                logfire.info('reranked search results: {results}', 
                    results=[(r.document, r.score) for r in reranked_results])
                
                final_results = reranked_results
                
            except Exception as e:
                # Fallback to embedding results if reranking fails
                LOG.error(f"Reranking failed, falling back to embedding results: {e}")
                final_results = [
                    r for r in initial_results 
                    if r.score >= ctx.deps.config.chat.similarity_threshold
                ]

            # Build context from final results
            ctx.state.retrieved_context = RetrievedContext(
                chunks=[r.document for r in final_results],
                metadatas=[r.metadata for r in final_results],
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
    vector_store: VectorStore,
    config: AppConfig
) -> ChatResult:
    """
    Process a chat question through the graph.
    This is the main entry point for chat processing.
    """
    state = ChatGraphState(original_question=question)
    deps = ChatGraphDeps(vector_store=vector_store, config=config)
    
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
    vector_store: VectorStore,
    config: AppConfig
) -> AsyncGenerator[StreamingChatResult, None]:
    """
    Stream chat progress through the graph.
    This is the main entry point for chat processing.
    """
    state = ChatGraphState(original_question=question)
    deps = ChatGraphDeps(vector_store=vector_store, config=config)
    
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

    