from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import chromadb
import numpy as np
from pydantic import BaseModel
from dataclasses import dataclass
from know_lang_bot.chat_bot.chat_graph import ChatResult
from know_lang_bot.config import AppConfig, EmbeddingConfig
import json
from know_lang_bot.evaluation.chatbot_evaluation import EvalCase, TRANSFORMER_TEST_CASES
from know_lang_bot.models.embeddings import generate_embedding, EmbeddingVector

@dataclass
class ConfigEvalResult:
    """Results for a single configuration"""
    config_name: str
    distances: List[float]
    stats: Dict[str, float]

class RetrievalMetrics(BaseModel):
    """Metrics for retrieval analysis"""
    distances: List[float]
    similarity_scores: List[float]
    chunk_count: int

class EnhancedChatResult(ChatResult):
    """Extended ChatResult with retrieval metrics"""
    retrieval_metrics: Optional[RetrievalMetrics] = None

def _embedding_cache_path(embdding_config: EmbeddingConfig) -> Path:
    """Get the cache path for a specific embedding configuration"""
    return Path(f"embeddings_{embdding_config.model_name}_{embdding_config.model_provider.value}.json")

def load_cached_embeddings(cache_path: Path, embdding_config: EmbeddingConfig) -> Optional[Dict[str, EmbeddingVector]]:
    """Load cached embeddings for a specific configuration"""
    cache_file = cache_path / _embedding_cache_path(embdding_config)
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)
    return None

def save_cached_embeddings(cache_path: Path, config: EmbeddingConfig, embeddings: Dict[str, EmbeddingVector]):
    """Save embeddings to cache"""
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / _embedding_cache_path(config)
    with open(cache_file, 'w') as f:
        json.dump(embeddings, f)

async def analyze_embedding_distributions(
    test_cases: List[EvalCase],
    configs: List[Tuple[str, AppConfig]],
    cache_path : Path
) -> List[ConfigEvalResult]:
    """Analyze embedding distance distributions for multiple configurations"""
    results = []
    
    for config_name, config in configs:
        # Try to load cached embeddings first
        cached_embeddings = load_cached_embeddings(cache_path, config.embedding)
        
        if cached_embeddings is None:
            print(f"Generating new embeddings for {config_name}...")
            # Generate embeddings for all test cases
            questions = [case.question for case in test_cases]
            try:
                embeddings = generate_embedding(questions, config.embedding)
                
                # Cache the embeddings
                cached_embeddings = {
                    question: embedding 
                    for question, embedding in zip(questions, embeddings)
                }
                save_cached_embeddings(cache_path, config.embedding, cached_embeddings)
            except Exception as e:
                print(f"Error generating embeddings for {config_name}: {str(e)}")
                continue
        else:
            print(f"Using cached embeddings for {config_name}")

        # Get collection for this config
        collection = chromadb.PersistentClient(
            path=str(config.db.persist_directory)
        ).get_collection(f"{config.db.collection_name}")
        
        distances = []
        
        # Query each test case
        for case in test_cases:
            query_results = collection.query(
                query_embeddings=[cached_embeddings[case.question]],
                n_results=10,
                include=['distances']
            )
            distances.extend(query_results['distances'][0])
        
        # Calculate statistics
        stats = {
            'mean': float(np.mean(distances)),
            'std': float(np.std(distances)),
            'median': float(np.median(distances)),
            'min': float(np.min(distances)),
            'max': float(np.max(distances))
        }
        
        results.append(ConfigEvalResult(
            config_name=config_name,
            distances=distances,
            stats=stats
        ))
    
    return results

def plot_distance_distributions(results: List[ConfigEvalResult]):
    """Plot and compare distance distributions for all configurations"""
    plt.figure(figsize=(15, 8))
    
    # Convert data to pandas DataFrame
    import pandas as pd
    all_data = pd.DataFrame([
        {'method': result.config_name, 'distance': d}
        for result in results
        for d in result.distances
    ])

    # Distribution plot
    plt.subplot(2, 2, 1)
    sns.histplot(
        data=all_data,
        x='distance',
        hue='method',
        stat='density',
        common_norm=True,
        alpha=0.6
    )
    plt.title('Distance Distribution Comparison')
    plt.xlabel('Cosine Distance')
    plt.ylabel('Density')
    
    # Box plot with pandas DataFrame
    plt.subplot(2, 2, 2)
    sns.boxplot(
        data=all_data,
        x='method',
        y='distance'
    )
    plt.title('Distance Distribution Statistics')
    plt.ylabel('Cosine Distance')
    plt.xticks(rotation=45)
    
    # Statistics summary
    plt.subplot(2, 2, (3, 4))
    stats_text = "Distance Statistics:\n\n"
    for result in results:
        stats_text += f"{result.config_name}:\n"
        stats_text += f"  Mean: {result.stats['mean']:.3f} ± {result.stats['std']:.3f}\n"
        stats_text += f"  Median: {result.stats['median']:.3f}\n"
        stats_text += f"  Range: [{result.stats['min']:.3f}, {result.stats['max']:.3f}]\n\n"
    
    plt.text(0.1, 0.9, stats_text, 
             transform=plt.gca().transAxes,
             verticalalignment='top',
             fontfamily='monospace')
    plt.axis('off')

    print(stats_text)
    
    plt.tight_layout()
    plt.savefig('embedding_comparison.png')
    plt.close()

async def main():
    # Define different configurations to compare
    configs = [
        ("Ollama Embedding", AppConfig(_env_file=Path('.env.evaluation.ollama'))),
        ("OpenAI Embedding", AppConfig(_env_file=Path('.env.evaluation.openai'))),
        # Add more configurations as needed
    ]
    
    # Analyze distributions for all configs
    results = await analyze_embedding_distributions(
        TRANSFORMER_TEST_CASES,
        configs,
        cache_path=Path("evaluations", "embedding_cache")
    )
    
    # Plot distributions
    plot_distance_distributions(results)
    
    return results

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())