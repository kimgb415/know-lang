import gradio as gr
from know_lang_bot.config import AppConfig
from know_lang_bot.utils.fancy_log import FancyLogger
from know_lang_bot.chat_bot.chat_graph import ChatResult, process_chat
import chromadb
from typing import List, Dict
import logfire
from pathlib import Path

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
    
    @logfire.instrument('Chatbot Process Question with {message=}')
    async def process_question(
        self,
        message: str,
        history: List[Dict[str, str]]
    ) -> ChatResult:
        """Process a question and return the answer with references"""
        return await process_chat(message, self.collection, self.config)
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface"""
        with gr.Blocks() as interface:
            gr.Markdown(f"# {self.config.chat.interface_title}")
            gr.Markdown(self.config.chat.interface_description)
            
            chatbot = gr.Chatbot(
                type="messages",
                bubble_full_width=False,
                render_markdown=True
            )
            
            msg = gr.Textbox(
                label="Ask about the codebase",
                placeholder="What does the CodeParser class do?",
                container=False
            )
            
            with gr.Row():
                submit = gr.Button("Submit")
                clear = gr.ClearButton([msg, chatbot])

            async def respond(message, history):
                result = await self.process_question(message, history)
                
                # Format the answer with code blocks
                formatted_messages = []
                
                # Add user message
                formatted_messages.append({
                    "role": "user", 
                    "content": message
                })
                
                # Collect code blocks first
                code_blocks = []
                if result.retrieved_context and result.retrieved_context.metadatas:
                    for metadata in result.retrieved_context.metadatas:
                        file_path = metadata['file_path']
                        start_line = metadata['start_line']
                        end_line = metadata['end_line']
                        
                        code = self._get_code_block(file_path, start_line, end_line)
                        if code:
                            title = f"📄 {file_path} (lines {start_line}-{end_line})"
                            if metadata.get('name'):
                                title += f" - {metadata['type']}: {metadata['name']}"
                            
                            code_blocks.append({
                                "role": "assistant",
                                "content": f"<details><summary>{title}</summary>\n\n```python\n{code}\n```\n\n</details>",
                            })
                
                # Add code blocks before the answer
                formatted_messages.extend(code_blocks)
                
                # Add assistant's answer
                formatted_messages.append({
                    "role": "assistant",
                    "content": result.answer
                })
                
                return {
                    msg: "",
                    chatbot: history + formatted_messages
                }

            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            submit.click(respond, [msg, chatbot], [msg, chatbot])

        return interface

def create_chatbot(config: AppConfig) -> gr.Blocks:
    interface = CodeQAChatInterface(config)
    return interface.create_interface()