import requests
import zipfile
import io
from know_lang_bot.chat_bot.chat_interface import create_chatbot
from know_lang_bot.configs.config import AppConfig
import tempfile
from rich.console import Console

console = Console()

# gradio demo
with tempfile.TemporaryDirectory() as temp_dir:
    config = AppConfig()
    with console.status("Downloading and extracting codebase...") as status:
        config.db.codebase_directory = temp_dir
        status.update(f"Downloading codebase into {temp_dir}...\n")
        

        # Download and unzip the code
        url = "https://github.com/huggingface/transformers/archive/refs/tags/v4.48.1.zip"
        response = requests.get(url)
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
        status.update(f"Extracting codebase...\n")

        # Extract the zip file into the codebase directory
        zip_file.extractall(config.db.codebase_directory)

    # Create and launch the chatbot
    demo = create_chatbot(config)
    demo.launch(server_name="0.0.0.0", server_port=7860)