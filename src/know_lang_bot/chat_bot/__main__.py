from know_lang_bot.config import AppConfig
from know_lang_bot.chat_bot.chat_graph import process_chat
import chromadb
import asyncio

async def test_chat_processing():
    config = AppConfig()
    db_client = chromadb.PersistentClient(
        path=str(config.db.persist_directory)
    )
    collection = db_client.get_collection(
        name=config.db.collection_name
    )
    
    result = await process_chat(
        "How are different quantization methods implemented in the transformers library, and what are the key components required to implement a new quantization method?",
        collection,
        config
    )
    
    print(f"Answer: {result.answer}")

if __name__ == "__main__":
    asyncio.run(test_chat_processing())