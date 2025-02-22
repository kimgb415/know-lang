import asyncio
from pathlib import Path
from typing import List

import chromadb
from chromadb.errors import InvalidCollectionException
from rich.console import Console
from rich.progress import Progress

from knowlang.configs import AppConfig, EmbeddingConfig
from knowlang.models import EmbeddingInputType, generate_embedding
from knowlang.utils import FancyLogger

LOG = FancyLogger(__name__)
console = Console()

BATCH_SIZE = 64  # VoyageAI's maximum batch size is 128

async def process_batch(
    documents: List[str],
    config: EmbeddingConfig,
) -> List[List[float]]:
    """Process a batch of documents to generate embeddings"""
    try:
        embeddings = generate_embedding(
            input=documents,
            config=config,
            input_type=EmbeddingInputType.DOCUMENT
        )
        return embeddings
    except Exception as e:
        LOG.error(f"Error processing batch: {e}")
        raise

async def migrate_embeddings(config: AppConfig):
    """Migrate embeddings using VoyageAI's API"""
    # Initialize source DB client (existing)
    source_client = chromadb.PersistentClient(
        path=str(config.db.persist_directory)
    )
    source_collection = source_client.get_collection(
        name=config.db.collection_name
    )
    
    # Initialize target DB client (new)
    target_path = Path(config.db.persist_directory).parent / f"transformers-{config.embedding.model_provider.value}-{config.embedding.model_name}"
    target_path.mkdir(exist_ok=True)
    target_client = chromadb.PersistentClient(path=str(target_path))
    
    # Create new collection
    new_collection_name = f"{config.db.collection_name}_voyage"
    try:
        target_collection = target_client.get_collection(name=new_collection_name)
        console.print(f"[yellow]Collection {new_collection_name} already exists. Deleting...")
        target_client.delete_collection(name=new_collection_name)
    except InvalidCollectionException:
        pass
    
    target_collection = target_client.create_collection(
        name=new_collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Get all documents from source
    results = source_collection.get(
        include=['documents', 'metadatas']
    )
    
    total_documents = len(results['ids'])
    console.print(f"[green]Found {total_documents} documents to process")
    
    with Progress() as progress:
        batch_task = progress.add_task(
            "Processing batches...", 
            total=total_documents
        )
        
        # Process in batches
        for i in range(0, total_documents, BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, total_documents)
            batch_docs = results['documents'][i:batch_end]
            batch_ids = results['ids'][i:batch_end]
            batch_metadatas = results['metadatas'][i:batch_end]
            
            try:
                # Generate embeddings for batch
                embeddings = await process_batch(
                    documents=batch_docs,
                    config=config.embedding
                )
                
                # Add to new collection
                target_collection.add(
                    embeddings=embeddings,
                    documents=batch_docs,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
            
                await asyncio.sleep(2)
                
            except Exception as e:
                LOG.error(f"Failed to process batch {i//BATCH_SIZE}: {e}")
                # Log failed IDs for retry
                failed_ids = batch_ids
                console.print(f"[red]Failed IDs: {failed_ids}")
                continue
            
            finally:
                progress.advance(batch_task, len(batch_docs))
    
    # Print statistics
    final_count = len(target_collection.get()['ids'])
    console.print(f"\n[green]Migration complete!")
    console.print(f"Source documents: {total_documents}")
    console.print(f"Target documents: {final_count}")
    console.print(f"\nNew database location: {target_path}")
    
    if final_count < total_documents:
        console.print(f"[yellow]Warning: {total_documents - final_count} documents failed to process")

if __name__ == "__main__":
    config = AppConfig()
    asyncio.run(migrate_embeddings(config))