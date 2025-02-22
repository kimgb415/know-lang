from knowlang.chat_bot.chat_interface import create_chatbot
from knowlang.configs import AppConfig

config = AppConfig()
demo = create_chatbot(config)
demo.launch(server_name="0.0.0.0", server_port=7860)