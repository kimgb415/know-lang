from pydantic_ai import Agent
from knowlang.configs import AppConfig
import logging
from knowlang.core.types import CodeChunk
from knowlang.utils import (
    create_pydantic_model,
    format_code_summary
)
from knowlang.vector_stores.factory import VectorStoreFactory

LOG = logging.getLogger(__name__)


class IndexingAgent:
    def __init__(
        self, 
        config: AppConfig,
    ):
        """
        Initialize IndexingAgent with config and optional vector store.
        If vector store is not provided, creates one based on config.
        """
        self.config = config
        self.vector_store = VectorStoreFactory.get(config.db, config.embedding)
        self._init_agent()

    def _init_agent(self):
        """Initialize the LLM agent with configuration"""
        system_prompt = """
You are an expert code analyzer specializing in creating searchable and contextual code summaries. 
Your summaries will be used in a RAG system to help developers understand complex codebases.
Focus on following points:
1. The main purpose and functionality
- Use precise technical terms
- Preserve class/function/variable names exactly
- State the primary purpose
2. Narrow down key implementation details
- Focus on key algorithms, patterns, or design choices
- Highlight important method signatures and interfaces
3. Any notable dependencies or requirements
- Reference related classes/functions by exact name
- List external dependencies
- Note any inherited or implemented interfaces
        
Provide a clean, concise and focused summary. Don't include unnecessary nor generic details.
"""
        
        self.agent = Agent(
            create_pydantic_model(
                model_provider=self.config.llm.model_provider,
                model_name=self.config.llm.model_name
            ),
            system_prompt=system_prompt,
            model_settings=self.config.llm.model_settings
        )

    async def summarize_chunk(self, chunk: CodeChunk) -> str:
        """Summarize a single code chunk using the LLM"""
        prompt = f"""
        Analyze this {chunk.type.value} code chunk:
        
        {chunk.content}
        
        {f'Docstring: {chunk.docstring}' if chunk.docstring else ''}
        
        Provide a concise summary.
        """
        
        result = await self.agent.run(prompt)

        return format_code_summary(chunk.content, result.data)