from dataclasses import dataclass
import gradio as gr
from knowlang.configs.chat_config import ChatConfig
from knowlang.configs.config import AppConfig
from knowlang.utils.fancy_log import FancyLogger
from knowlang.utils.rate_limiter import RateLimiter
from knowlang.chat_bot.chat_graph import stream_chat_progress, ChatStatus
from knowlang.chat_bot.feedback import ChatAnalytics
import chromadb
from typing import List, Dict, AsyncGenerator
from pathlib import Path
from gradio import ChatMessage


LOG = FancyLogger(__name__)

@dataclass
class CodeContext:
    file_path: str
    start_line: int
    end_line: int

    def to_title(self, config: ChatConfig) -> str:
        """Format code context as a title string"""
        truncated_file_path = self.file_path[len(config.code_path_prefix):]
        title = f"📄 {truncated_file_path} (lines {self.start_line}-{self.end_line})"
        return title
    
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
        self._init_chroma()
        self.codebase_dir = Path(config.db.codebase_directory)
        self.rate_limiter = RateLimiter()
        self.chat_analytics = ChatAnalytics(config.chat_analytics)
        
    def _init_chroma(self):
        """Initialize ChromaDB connection"""
        self.db_client = chromadb.PersistentClient(
            path=str(self.config.db.persist_directory)
        )
        self.collection = self.db_client.get_collection(
            name=self.config.db.collection_name
        )
    
    def _get_code_block(self, file_path: str, start_line: int, end_line: int) -> str:
        """Read the specified lines from a file and return as a code block"""
        try:
            full_path = self.codebase_dir / file_path
            with open(full_path, 'r') as f:
                lines = f.readlines()
                code_lines = lines[start_line-1:end_line]
                return ''.join(code_lines)
        except Exception as e:
            LOG.error(f"Error reading code block: {e}")
            return "Error reading code"

    def _format_code_block(self, metadata: Dict) -> str:
        """Format a single code block with metadata"""
        context = CodeContext.from_metadata(metadata)
        code = self._get_code_block(
            context.file_path, 
            context.start_line, 
            context.end_line
        )
        if not code:
            return None

        return f"<details><summary>{context.to_title(self.config.chat)}</summary>\n\n```python\n{code}\n```\n\n</details>"
    
    def _handle_feedback(self, like_data: gr.LikeData, history: List[ChatMessage], request: gr.Request):
         # Get the query and response pair
        query = history[like_data.index - 1]["content"]  # User question
        
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

        # Check rate limit before processing
        client_ip : str = request.request.client.host
        print(f"Client IP: {client_ip}")
        if self.rate_limiter.check_rate_limit(client_ip):
            wait_time = self.rate_limiter.get_remaining_time(client_ip)
            rate_limit_message = (
                f"Rate limit exceeded. Please wait {wait_time:.0f} seconds before sending another message."
            )
            history.append(ChatMessage(
                role="assistant",
                content=rate_limit_message,
                metadata={
                    "title": "⚠️ Rate Limit Warning",
                    "status": "done"
                }
            ))
            yield history
            return
        
        current_progress: ChatMessage | None = None
        code_blocks_added = False
        
        async for result in stream_chat_progress(message, self.collection, self.config):
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
                for metadata in result.retrieved_context.metadatas:
                    code_block = self._format_code_block(metadata)
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
                "How does Trainer handle distributed training and gradient accumulation? Explain the implementation details.",
                "How does the text generation pipeline handle chat-based generation and template processing?",
                "How does the transformers library automatically select and configure the appropriate quantization method?",
                "How to implement top-k filtering for text generation?"
            ]
            
            msg = gr.Textbox(
                label="Ask about the codebase",
                placeholder="what are the key components required to implement a new quantization method?",
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

            async def respond(message: str, history: List[ChatMessage], request: gr.Request) -> AsyncGenerator[List[ChatMessage], None]:
                self.chat_analytics.track_query(message, request.request.client.host)
                async for updated_history in self.stream_response(message, history, request):
                    yield updated_history
                        
            # Set up event handlers
            msg.submit(respond, [msg, chatbot], [chatbot])
            submit.click(respond, [msg, chatbot], [chatbot])
            chatbot.like(self._handle_feedback, [chatbot])

        return interface

def create_chatbot(config: AppConfig) -> gr.Blocks:
    interface = CodeQAChatInterface(config)
    return interface.create_interface()