from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncIterator, Optional
from contextlib import asynccontextmanager

from pydantic_ai.models import (
    AgentModel, 
    ModelMessage, 
    ModelResponse, 
    StreamedResponse, 
    Usage, 
    Model,
    check_allow_model_requests
)
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.messages import (
    ModelResponsePart, TextPart,  SystemPromptPart,
    UserPromptPart, ToolReturnPart, ModelResponseStreamEvent
)
from pydantic_ai.settings import ModelSettings
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch


@dataclass(init=False)
class HuggingFaceModel(Model):
    """A model that uses HuggingFace models locally.
    
    For MVP, this implements basic functionality without streaming or advanced features.
    """
    
    model_name: str
    model: AutoModelForCausalLM = field(repr=False)
    tokenizer: AutoTokenizer = field(repr=False)
    device: str = field(default="cuda" if torch.cuda.is_available() else "cpu")
    
    def __init__(
        self,
        model_name: str,
        *,
        device: Optional[str] = None,
        max_new_tokens: int = 512,
    ):
        """Initialize a HuggingFace model.
        
        Args:
            model_name: Name of the model on HuggingFace Hub
            device: Device to run model on ('cuda' or 'cpu')
            max_new_tokens: Maximum number of tokens to generate
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        """Create an agent model for each step of an agent run."""
        check_allow_model_requests()
        
        return HuggingFaceAgentModel(
            model=self.model,
            tokenizer=self.tokenizer,
            model_name=self.model_name,
            device=self.device,
            max_new_tokens=self.max_new_tokens,
            tools=function_tools if function_tools else None
        )

    def name(self) -> str:
        return f"huggingface:{self.model_name}"

@dataclass
class HuggingFaceAgentModel(AgentModel):
    """Implementation of AgentModel for HuggingFace models."""
    
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    model_name: str
    device: str
    max_new_tokens: int
    tools: Optional[list[ToolDefinition]] = None
    
    def _format_messages(self, messages: list[ModelMessage]) -> str:
        """Format messages into a prompt the model can understand."""
        formatted = []
        for message in messages:
            for part in message.parts:
                if isinstance(part, SystemPromptPart):
                    formatted.append(f"<|system|>{part.content}")
                elif isinstance(part, UserPromptPart):
                    formatted.append(f"<|user|>{part.content}")
                else:
                    # For MVP, we'll just pass through other message types
                    formatted.append(str(part.content))
        formatted.append("<|assistant|>")
        return "\n".join(formatted)

    async def request(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None
    ) -> tuple[ModelResponse, Usage]:
        """Make a request to the model."""
        prompt = self._format_messages(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = response_text[len(prompt):]  # Remove the prompt
        
        timestamp = datetime.now(timezone.utc)
        response = ModelResponse(
            parts=[TextPart(response_text)],
            model_name=self.model_name,
            timestamp=timestamp
        )
        
        usage_stats = Usage(
            requests=1,
            request_tokens=inputs.input_ids.shape[1],
            response_tokens=len(outputs[0]) - inputs.input_ids.shape[1],
            total_tokens=len(outputs[0])
        )
        
        return response, usage_stats

    @asynccontextmanager
    async def request_stream(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None
    ) -> AsyncIterator[StreamedResponse]:
        """Make a streaming request to the model."""
        prompt = self._format_messages(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        
        generation_kwargs = dict(
            input_ids=inputs.input_ids,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            streamer=streamer,
        )
        
        thread = torch.jit.fork(self.model.generate, **generation_kwargs)
        
        try:
            yield HuggingFaceStreamedResponse(
                _model_name=self.model_name,
                _streamer=streamer,
                _timestamp=datetime.now(timezone.utc),
            )
        finally:
            torch.jit.wait(thread)

@dataclass 
class HuggingFaceStreamedResponse(StreamedResponse):
    """Implementation of StreamedResponse for HuggingFace models."""
    
    _streamer: TextIteratorStreamer
    _timestamp: datetime

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Stream tokens from the model."""
        for new_text in self._streamer:
            self._usage.response_tokens += len(new_text)
            self._usage.total_tokens = self._usage.request_tokens + self._usage.response_tokens
            
            yield self._parts_manager.handle_text_delta(
                vendor_part_id='content',
                content=new_text
            )

    def timestamp(self) -> datetime:
        return self._timestamp