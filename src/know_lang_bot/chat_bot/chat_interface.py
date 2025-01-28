import gradio as gr
from know_lang_bot.chat_bot.chat_config import ChatAppConfig, chat_app_config
from know_lang_bot.utils.fancy_log import FancyLogger
from know_lang_bot.chat_bot.chat_graph import ChatResult, process_chat
import chromadb
from typing import List, Dict
import logfire

LOG = FancyLogger(__name__)

class CodeQAChatInterface:
    def __init__(self, config: ChatAppConfig):
        self.config = config
        self._init_chroma()
        
    def _init_chroma(self):
        """Initialize ChromaDB connection"""
        self.db_client = chromadb.PersistentClient(
            path=str(self.config.db.persist_directory)
        )
        self.collection = self.db_client.get_collection(
            name=self.config.db.collection_name
        )
    
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
            
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(type="messages", bubble_full_width=False)
                    msg = gr.Textbox(
                        label="Ask about the codebase",
                        placeholder="What does the CodeParser class do?",
                        container=False
                    )
                    clear = gr.ClearButton([msg, chatbot])
                
                with gr.Column(scale=1):
                    references = gr.Markdown(
                        label="Referenced Code",
                        value="Code references will appear here..."
                    )

            async def respond(message, history):
                result = await self.process_question(message, history)
                references.value = result.references_md
                return {
                    msg: "",
                    chatbot: history + [
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": result.answer}
                    ]
                }

            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            clear.click(lambda: [], None, references)

        return interface

def create_chatbot() -> gr.Blocks:
    interface = CodeQAChatInterface(chat_app_config)

    return interface.create_interface()