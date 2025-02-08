import asyncio
import json
from pathlib import Path
import chromadb
from chromadb.errors import InvalidCollectionException
from rich.progress import Progress
from rich.console import Console
from typing import List, Dict, Optional
from openai import OpenAI
from datetime import datetime
from knowlang.configs.config import AppConfig
from knowlang.utils.fancy_log import FancyLogger
from knowlang.utils.chunking_util import truncate_chunk

LOG = FancyLogger(__name__)
console = Console()

BATCH_SIZE = 2000  # Max items per batch

class BatchState:
    """Class to track batch processing state"""
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.batch_dir = root_dir / "batches"
        self.results_dir = root_dir / "results"
        self.metadata_dir = root_dir / "metadata"
        
        # Create directories
        for dir in [self.batch_dir, self.results_dir, self.metadata_dir]:
            dir.mkdir(parents=True, exist_ok=True)
    
    def save_batch_metadata(self, batch_id: str, metadata: Dict):
        """Save batch processing metadata"""
        with open(self.metadata_dir / f"{batch_id}.json", "w") as f:
            json.dump(metadata, f, indent=2)



async def prepare_batches(config: AppConfig, batch_state: BatchState) -> List[str]:
    """Prepare batch files from ChromaDB and return batch IDs"""
    source_client = chromadb.PersistentClient(path=str(config.db.persist_directory))
    source_collection = source_client.get_collection(name=config.db.collection_name)
    
    # Get all documents
    results = source_collection.get(include=['documents', 'metadatas' ])

    if not results['ids']:
        console.print("[red]No documents found in source collection!")
        return
    total_documents = len(results['ids'])
    
    batch_ids = []
    with Progress() as progress:
        task = progress.add_task("Preparing batches...", total=total_documents)
        
        current_batch = []
        current_batch_ids = []
        current_batch_num = 0
        
        for i, (doc_id, doc, metadata) in enumerate(zip(
            results['ids'], 
            results['documents'], 
            results['metadatas']
        )):
            # Truncate document if needed
            truncated_doc = truncate_chunk(doc)
            
            current_batch.append((doc_id, truncated_doc))
            current_batch_ids.append(doc_id)
            
            # Create batch file when size limit reached or at end
            if len(current_batch) >= BATCH_SIZE or i == total_documents - 1:
                batch_file = batch_state.batch_dir / f"batch_{current_batch_num}.jsonl"
                
                with open(batch_file, 'w') as f:
                    for bid, bdoc in current_batch:
                        request = {
                            "custom_id": bid,
                            "method": "POST",
                            "url": "/v1/embeddings",
                            "body": {
                                "model": config.embedding.model_name,
                                "input": bdoc
                            }
                        }
                        f.write(json.dumps(request) + '\n')
                
                # Save batch metadata
                batch_metadata = {
                    "batch_id": f"batch_{current_batch_num}",
                    "created_at": datetime.now().isoformat(),
                    "document_ids": current_batch_ids,
                    "size": len(current_batch),
                    "status": "prepared"
                }
                batch_state.save_batch_metadata(f"batch_{current_batch_num}", batch_metadata)
                
                batch_ids.append(f"batch_{current_batch_num}")
                current_batch = []
                current_batch_ids = []
                current_batch_num += 1
            
            progress.advance(task)
    
    return batch_ids

async def submit_batches(batch_state: BatchState, batch_ids: List[str]):
    """Submit prepared batches to OpenAI"""
    client = OpenAI()
    
    with Progress() as progress:
        task = progress.add_task("Submitting batches...", total=len(batch_ids))
        
        for batch_id in batch_ids:
            batch_file = batch_state.batch_dir / f"{batch_id}.jsonl"
            
            # Upload batch file
            file = client.files.create(
                file=open(batch_file, "rb"),
                purpose="batch"
            )
            
            # Create batch job
            batch = client.batches.create(
                input_file_id=file.id,
                endpoint="/v1/embeddings",
                completion_window="24h"
            )
            
            # Update metadata
            with open(batch_state.metadata_dir / f"{batch_id}.json", "r") as f:
                metadata = json.load(f)
            
            metadata.update({
                "openai_batch_id": batch.id,
                "file_id": file.id,
                "status": "submitted",
                "submitted_at": datetime.now().isoformat()
            })
            
            batch_state.save_batch_metadata(batch_id, metadata)
            progress.advance(task)

async def process_batch_results(
    batch_state: BatchState,
    config: AppConfig,
    batch_ids: Optional[List[str]] = None
):
    """Process completed batches and store in new ChromaDB"""
    client = OpenAI()
    
    # Initialize target DB
    target_path = Path(config.db.persist_directory).parent / "batch_embeddings_db"
    target_path.mkdir(exist_ok=True)
    target_client = chromadb.PersistentClient(path=str(target_path))
    
    # Create or get collection
    new_collection_name = f"{config.db.collection_name}_batch"
    try:
        target_collection = target_client.get_collection(name=new_collection_name)
        console.print(f"[yellow]Collection {new_collection_name} exists, appending...")
    except InvalidCollectionException:
        target_collection = target_client.create_collection(
            name=new_collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    # Process each batch
    if batch_ids is None:
        batch_ids = [f.stem for f in batch_state.metadata_dir.glob("*.json")]
    
    with Progress() as progress:
        task = progress.add_task("Processing results...", total=len(batch_ids))
        
        for batch_id in batch_ids:
            # Load batch metadata
            with open(batch_state.metadata_dir / f"{batch_id}.json", "r") as f:
                metadata = json.load(f)
            
            if metadata["status"] != "submitted":
                console.print(f"[yellow]Skipping {batch_id} - not submitted")
                progress.advance(task)
                continue
            
            # Check batch status
            batch_status =  client.batches.retrieve(metadata["openai_batch_id"])
            if batch_status.status != "completed":
                console.print(f"[yellow]Batch {batch_id} not complete, status: {batch_status.status}")
                progress.advance(task)
                continue
            
            # Download results
            output_file = batch_state.results_dir / f"{batch_id}_output.jsonl"
            response = client.files.content(batch_status.output_file_id)
            with open(output_file, "wb") as f:
                f.write(response.read())
            
            # Process embeddings
            source_client = chromadb.PersistentClient(path=str(config.db.persist_directory))
            source_collection = source_client.get_collection(name=config.db.collection_name)
            
            # Get original documents and metadata
            results = source_collection.get(
                ids=metadata["document_ids"],
                include=['documents', 'metadatas']
            )
            
            # Process results file
            embeddings = []
            processed_ids = []
            processed_docs = []
            processed_metadatas = []
            
            with open(output_file) as f:
                for line in f:
                    result = json.loads(line)
                    if result["response"]["status_code"] == 200:
                        doc_idx = metadata["document_ids"].index(result["custom_id"])
                        
                        embeddings.append(result["response"]["body"]["data"][0]["embedding"])
                        processed_ids.append(result["custom_id"])
                        processed_docs.append(results["documents"][doc_idx])
                        processed_metadatas.append(results["metadatas"][doc_idx])
            
            # Add to new collection
            if processed_ids:
                target_collection.add(
                    embeddings=embeddings,
                    documents=processed_docs,
                    metadatas=processed_metadatas,
                    ids=processed_ids
                )
            
            # Update metadata
            metadata.update({
                "status": "processed",
                "processed_at": datetime.now().isoformat(),
                "processed_count": len(processed_ids)
            })
            batch_state.save_batch_metadata(batch_id, metadata)
            
            progress.advance(task)

async def main():
    config = AppConfig()
    batch_state = BatchState(Path("embedding_migration"))
    
    # Step 1: Prepare batches
    console.print("[green]Step 1: Preparing batches...")
    batch_ids = await prepare_batches(config, batch_state)
    
    # # Step 2: Submit batches
    console.print("\n[green]Step 2: Submitting batches...")
    await submit_batches(batch_state, batch_ids)
    
    # Step 3: Process results
    console.print("\n[green]Step 3: Processing results...")
    await process_batch_results(batch_state, config)

if __name__ == "__main__":
    asyncio.run(main())