from know_lang_bot.code_parser.parser import CodeParser
from know_lang_bot.config import AppConfig
from know_lang_bot.code_parser.summarizer import CodeSummarizer
import asyncio

# Usage example
if __name__ == "__main__":
    async def main():
        config = AppConfig()  # Will load from .env file if available
        summarizer = CodeSummarizer(config)
        
        # Example usage with your parser
        parser = CodeParser(".")
        chunks = parser.parse_repository()
        
        await summarizer.process_chunks(chunks)
    
    asyncio.run(main())