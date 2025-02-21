from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List

import gradio as gr
from gradio import ChatMessage

from knowlang.chat_bot import ChatAnalytics, ChatStatus, stream_chat_progress
from knowlang.configs.config import AppConfig
from knowlang.utils import FancyLogger, RateLimiter
from knowlang.vector_stores.factory import VectorStoreFactory

LOG = FancyLogger(__name__)

@dataclass
class CodeContext:
    file_path: str
    start_line: int
    end_line: int

    def to_title(self) -> str:
        """Format code context as a title string"""
        return f"📄 {self.file_path} (lines {self.start_line}-{self.end_line})"
    
    @classmethod
    def from_metadata(cls, metadata: Dict) -> "CodeContext":
        """Create code context from metadata dictionary"""
        return cls(
            file_path=metadata['file_path'],
            start_line=metadata['start_line'],
            end_line=metadata['end_line'],
        )

class CodeQAChatInterface:
    def __init__(self, config: AppConfig):
        self.config = config
        self.vector_store = VectorStoreFactory.get(config.db)
        self.rate_limiter = RateLimiter()
        self.chat_analytics = ChatAnalytics(config.chat_analytics)
    
    def _format_code_block(self, code : str,  metadata: Dict) -> str:
        """Format a single code block with metadata"""
        context = CodeContext.from_metadata(metadata)

        return f"<details><summary>{context.to_title()}</summary>\n\n```python\n{code}\n```\n\n</details>"
    
    def _handle_feedback(self, like_data: gr.LikeData, history: List[ChatMessage], request: gr.Request):
         # Get the query and response pair
        query = history[like_data.index - 1].content  # User question
        
        # Track feedback
        self.chat_analytics.track_feedback(
            like=like_data.liked,  # True for thumbs up, False for thumbs down
            query=query,
            client_ip=request.request.client.host
        )


    async def stream_response(
        self,
        message: str,
        history: List[ChatMessage],
        request: gr.Request, # gradio injects the request object
    ) -> AsyncGenerator[List[ChatMessage], None]:
        """Stream chat responses with progress updates"""
        # Add user message
        history.append(ChatMessage(role="user", content=message))
        yield history

        
        current_progress: ChatMessage | None = None
        code_blocks_added = False
        
        try:
            async for result in stream_chat_progress(message, self.vector_store, self.config):
                # Handle progress updates
                if result.status != ChatStatus.COMPLETE:
                    if current_progress:
                        history.remove(current_progress)
                    
                    current_progress = ChatMessage(
                        role="assistant",
                        content=result.progress_message,
                        metadata={
                            "title": f"{result.status.value.title()} Progress",
                            "status": "pending" if result.status != ChatStatus.ERROR else "error"
                        }
                    )
                    history.append(current_progress)
                    yield history
                    continue

                # When complete, remove progress message and add final content
                if current_progress:
                    history.remove(current_progress)
                    current_progress = None

                # Add code blocks before final answer if not added yet
                if not code_blocks_added and result.retrieved_context and result.retrieved_context.metadatas:
                    total_code_blocks = []
                    for chunk, metadata in zip(result.retrieved_context.chunks, result.retrieved_context.metadatas):
                        code_block = self._format_code_block(chunk, metadata)
                        if code_block:
                            total_code_blocks.append(code_block)

                    code_blocks_added = True
                    history.append(ChatMessage(
                        role="assistant",
                        content='\n\n'.join(total_code_blocks),
                        metadata={
                            "title": "💻 Code Context",
                            "collapsible": True
                        }
                    ))
                    yield history

                # Add final answer
                history.append(ChatMessage(
                    role="assistant",
                    content=result.answer
                ))
                yield history
        except Exception as e:
            LOG.error(f"Error in stream_response: {e}")
            history.append(ChatMessage(
                role="assistant",
                content=f"An error occurred while processing your request: {str(e)}",
                metadata={
                    "title": "❌ Error",
                    "status": "error"
                }
            ))
            yield history

    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface"""
        with gr.Blocks() as interface:
            gr.Markdown(f"# {self.config.chat.interface_title}")
            gr.Markdown(self.config.chat.interface_description)
            
            chatbot = gr.Chatbot(
                type="messages",
                bubble_full_width=False,
                render_markdown=True,
                height=600
            )
            
            # Add example questions
            example_questions = [
                "what are the key components required to implement a new quantization method?",
                "How does Trainer handle distributed training and gradient accumulation? Explain the implementation details.",
                "How does the text generation pipeline handle chat-based generation and template processing?",
                "How does the transformers library automatically select and configure the appropriate quantization method?",
                "How to implement top-k filtering for text generation?"
            ]
            
            msg = gr.Textbox(
                label="Ask about the codebase",
                placeholder=self.config.chat.interface_placeholder,
                container=False,
                scale=7
            )

            gr.Examples(
                examples=example_questions,
                inputs=msg,
                label="Example Questions",
                examples_per_page=6
            )
            
            with gr.Row():
                submit = gr.Button("Submit", scale=1)
                clear = gr.ClearButton([msg, chatbot], scale=1)

            async def respond(message: str, history: List[ChatMessage], request: gr.Request) -> AsyncGenerator[tuple[List[ChatMessage],str], None]:
                self.chat_analytics.track_query(message, request.request.client.host)
                async for updated_history in self.stream_response(message, history, request):
                    # clear user input after processing
                    yield updated_history, ""
                        
            # Set up event handlers
            msg.submit(respond, [msg, chatbot], [chatbot, msg])
            submit.click(respond, [msg, chatbot], [chatbot, msg])
            chatbot.like(self._handle_feedback, [chatbot])

        return interface

def create_chatbot(config: AppConfig) -> gr.Blocks:
    interface = CodeQAChatInterface(config)
    return interface.create_interface()