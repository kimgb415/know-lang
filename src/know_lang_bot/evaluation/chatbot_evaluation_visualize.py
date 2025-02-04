from pathlib import Path
import json
import pandas as pd
from rich.console import Console
from rich.table import Table
from typing import List
from know_lang_bot.evaluation.chatbot_evaluation import EvalSummary

class ResultAnalyzer:
    def __init__(self, base_dir: Path):
        self.console = Console()
        self.embedding_dir = base_dir / "embedding"
        self.reranking_dir = base_dir / "embedding_reranking"
        
    def load_results(self, file_path: Path) -> List[EvalSummary]:
        """Load evaluation results from JSON file"""
        with open(file_path) as f:
            obj_list = json.load(f)
            return [EvalSummary.model_validate(obj) for obj in obj_list]
        
    def create_dataframe(self, results: List[EvalSummary]) -> pd.DataFrame:
        """Convert results to pandas DataFrame with flattened metrics"""
        rows = []
        for result in results:
            row = {
                "evaluator_model": result.evaluator_model,
                "question": result.case.question,
                "difficulty": result.case.difficulty,
                "chunk_relevance": result.eval_response.chunk_relevance,
                "answer_correctness": result.eval_response.answer_correctness,
                "code_reference": result.eval_response.code_reference,
                "weighted_total": result.eval_response.weighted_total
            }
            rows.append(row)
        
        return pd.DataFrame(rows)

    def analyze_results(self):
        """Analyze and display results comparison"""
        # Load all results
        all_results = {
            "embedding": [],
            "reranking": []
        }
        
        for file in self.embedding_dir.glob("*.json"):
            all_results["embedding"].extend(self.load_results(file))
        
        for file in self.reranking_dir.glob("*.json"):
            all_results["reranking"].extend(self.load_results(file))

        # Convert to DataFrames
        embedding_df = self.create_dataframe(all_results["embedding"])
        reranking_df = self.create_dataframe(all_results["reranking"])

        # Calculate statistics by evaluator model
        def get_model_stats(df: pd.DataFrame) -> pd.DataFrame:
            return df.groupby("evaluator_model").agg({
                "chunk_relevance": ["mean", "std"],
                "answer_correctness": ["mean", "std"],
                "code_reference": ["mean", "std"],
                "weighted_total": ["mean", "std"]
            }).round(2)

        embedding_stats = get_model_stats(embedding_df)
        reranking_stats = get_model_stats(reranking_df)

        # Display comparison tables
        self.display_comparison_table(embedding_stats, reranking_stats)
        self.display_improvement_metrics(embedding_df, reranking_df)
        
        # Save detailed results to CSV
        self.save_detailed_results(embedding_df, reranking_df)

    def display_comparison_table(self, embedding_stats: pd.DataFrame, reranking_stats: pd.DataFrame):
        """Display rich table comparing embedding and reranking results"""
        table = Table(title="Embedding vs Reranking Comparison")
        
        table.add_column("Metric", style="cyan")
        table.add_column("Model", style="magenta")
        table.add_column("Embedding", style="blue")
        table.add_column("Reranking", style="green")
        table.add_column("Improvement", style="yellow")

        metrics = ["chunk_relevance", "answer_correctness", "code_reference", "weighted_total"]
        
        for metric in metrics:
            for model in embedding_stats.index:
                emb_mean = embedding_stats.loc[model, (metric, "mean")]
                emb_std = embedding_stats.loc[model, (metric, "std")]
                rer_mean = reranking_stats.loc[model, (metric, "mean")]
                rer_std = reranking_stats.loc[model, (metric, "std")]
                
                improvement = ((rer_mean - emb_mean) / emb_mean * 100).round(1)
                
                table.add_row(
                    metric.replace("_", " ").title(),
                    model.split(":")[-1],
                    f"{emb_mean:.2f} ±{emb_std:.2f}",
                    f"{rer_mean:.2f} ±{rer_std:.2f}",
                    f"{improvement:+.1f}%" if improvement else "0%"
                )

        self.console.print(table)

    def display_improvement_metrics(self, embedding_df: pd.DataFrame, reranking_df: pd.DataFrame):
        """Display additional improvement metrics"""
        # Calculate improvements by difficulty
        difficulties = sorted(embedding_df["difficulty"].unique())
        
        table = Table(title="Improvements by Difficulty")
        table.add_column("Difficulty", style="cyan")
        table.add_column("Embedding", style="blue")
        table.add_column("Reranking", style="green")
        table.add_column("Improvement", style="yellow")

        for diff in difficulties:
            emb_score = embedding_df[embedding_df["difficulty"] == diff]["weighted_total"].mean()
            rer_score = reranking_df[reranking_df["difficulty"] == diff]["weighted_total"].mean()
            improvement = ((rer_score - emb_score) / emb_score * 100).round(1)
            
            table.add_row(
                str(diff),
                f"{emb_score:.2f}",
                f"{rer_score:.2f}",
                f"{improvement:+.1f}%"
            )

        self.console.print(table)

    def save_detailed_results(self, embedding_df: pd.DataFrame, reranking_df: pd.DataFrame):
        """Save detailed results to CSV"""
        # Add method column
        embedding_df["method"] = "embedding"
        reranking_df["method"] = "reranking"
        
        # Combine and save
        combined_df = pd.concat([embedding_df, reranking_df])
        combined_df.to_csv("evaluation_comparison.csv", index=False)
        self.console.print(f"\nDetailed results saved to evaluation_comparison.csv")

if __name__ == "__main__":
    analyzer = ResultAnalyzer(Path("evaluations"))
    analyzer.analyze_results()