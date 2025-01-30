import gradio as gr
from know_lang_bot.config import AppConfig
from know_lang_bot.utils.fancy_log import FancyLogger
from know_lang_bot.chat_bot.chat_graph import stream_chat_progress, ChatStatus
import chromadb
from typing import List, Dict, AsyncGenerator
from pathlib import Path
from gradio import ChatMessage


LOG = FancyLogger(__name__)

class CodeQAChatInterface:
    def __init__(self, config: AppConfig):
        self.config = config
        self._init_chroma()
        self.codebase_dir = Path(config.db.codebase_directory)
        
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
        file_path = metadata['file_path']
        start_line = metadata['start_line']
        end_line = metadata['end_line']
        
        code = self._get_code_block(file_path, start_line, end_line)
        if not code:
            return None
            
        title = f"📄 {file_path} (lines {start_line}-{end_line})"
        if metadata.get('name'):
            title += f" - {metadata['type']}: {metadata['name']}"
            

        return f"<details><summary>{title}</summary>\n\n```python\n{code}\n```\n\n</details>"

    async def stream_response(
        self,
        message: str,
        history: List[ChatMessage]
    ) -> AsyncGenerator[List[ChatMessage], None]:
        """Stream chat responses with progress updates"""
        # Add user message
        history.append(ChatMessage(role="user", content=message))
        yield history
        
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
            
            msg = gr.Textbox(
                label="Ask about the codebase",
                placeholder="What does the CodeParser class do?",
                container=False,
                scale=7
            )
            
            with gr.Row():
                submit = gr.Button("Submit", scale=1)
                clear = gr.ClearButton([msg, chatbot], scale=1)

            async def respond(message: str, history: List[ChatMessage]) -> AsyncGenerator[List[ChatMessage], None]:
                async for updated_history in self.stream_response(message, history):
                    yield updated_history
                    
            # Set up event handlers
            msg.submit(respond, [msg, chatbot], [chatbot])
            submit.click(respond, [msg, chatbot], [chatbot])

        return interface

def create_chatbot(config: AppConfig) -> gr.Blocks:
    interface = CodeQAChatInterface(config)
    return interface.create_interface()