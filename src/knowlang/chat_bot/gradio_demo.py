from knowlang.chat_bot.chat_interface import create_chatbot
from knowlang.configs.config import AppConfig


config = AppConfig()
demo = create_chatbot(config)
demo.launch()