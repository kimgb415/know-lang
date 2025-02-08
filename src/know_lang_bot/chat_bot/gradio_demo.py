from know_lang_bot.chat_bot.chat_interface import create_chatbot
from know_lang_bot.configs.config import AppConfig


config = AppConfig()
demo = create_chatbot(config)
demo.launch()