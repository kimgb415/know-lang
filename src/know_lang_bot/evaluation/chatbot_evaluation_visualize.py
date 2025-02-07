from pathlib import Path
import json
import pandas as pd
from rich.console import Console
from rich.table import Table
from typing import List, Dict
from enum import Enum
from know_lang_bot.evaluation.chatbot_evaluation import EvalSummary

class RetrievalMethod(str, Enum):
    EMBEDDING = "embedding"
    EMBEDDING_RERANKING = "embedding_reranking"
    EMBEDDING_WITH_CODE = "embedding_with_code"
    OPENAI_EMBEDDING_WITH_CODE = "openai_embedding_with_code"
    EMBEDDING_RERANKING_WITH_CODE = "embedding_reranking_with_code"

class ResultAnalyzer:
    def __init__(self, base_dir: Path, baseline_method: RetrievalMethod = RetrievalMethod.EMBEDDING):
        self.console = Console()
        self.base_dir = base_dir
        self.baseline_method = baseline_method
        # Map each method to its directory
        self.method_dirs = {
            RetrievalMethod.EMBEDDING: self.base_dir / RetrievalMethod.EMBEDDING.value,
            # RetrievalMethod.EMBEDDING_RERANKING: self.base_dir / RetrievalMethod.EMBEDDING_RERANKING.value,
            RetrievalMethod.EMBEDDING_WITH_CODE: self.base_dir / RetrievalMethod.EMBEDDING_WITH_CODE.value,
            RetrievalMethod.OPENAI_EMBEDDING_WITH_CODE: self.base_dir / RetrievalMethod.OPENAI_EMBEDDING_WITH_CODE.value,
            RetrievalMethod.VOYAGE_EMBEDDING_WITH_CODE: self.base_dir / RetrievalMethod.VOYAGE_EMBEDDING_WITH_CODE.value,
            # RetrievalMethod.EMBEDDING_RERANKING_WITH_CODE: self.base_dir / RetrievalMethod.EMBEDDING_RERANKING_WITH_CODE.value,
        }

    def load_results(self, file_path: Path) -> List[EvalSummary]:
        """Load evaluation results from JSON file"""
        with open(file_path) as f:
            obj_list = json.load(f)
            return [EvalSummary.model_validate(obj) for obj in obj_list]

    def create_dataframe(self, results: List[EvalSummary]) -> pd.DataFrame:
        """Convert results to pandas DataFrame with flattened metrics"""
        rows = []
        for result in results:
            # Basic metrics
            base_row = {
                "evaluator_model": result.evaluator_model,
                "question": result.case.question,
                "difficulty": result.case.difficulty,
                "environment": getattr(result.case, 'environment', 'default')
            }
            
            # Add metrics for each round
            for round in result.eval_rounds:
                row = base_row.copy()
                row.update({
                    "round_id": round.round_id,
                    "chunk_relevance": round.eval_response.chunk_relevance,
                    "answer_correctness": round.eval_response.answer_correctness,
                    "code_reference": round.eval_response.code_reference,
                    "weighted_total": round.eval_response.weighted_total,
                    "timestamp": round.timestamp
                })
                rows.append(row)
        
        return pd.DataFrame(rows)

    def load_all_results(self) -> Dict[RetrievalMethod, pd.DataFrame]:
        """Load results for all available methods"""
        results = {}
        for method in RetrievalMethod:
            method_dir = self.method_dirs.get(method)
            if method_dir and method_dir.exists():
                all_results = []
                for file in method_dir.glob("*.json"):
                    all_results.extend(self.load_results(file))
                if all_results:  # Only include methods with results
                    results[method] = self.create_dataframe(all_results)
        return results

    def calculate_improvement(self, new_val: float, baseline_val: float) -> str:
        """Calculate and format improvement percentage"""
        if baseline_val == 0:
            return "N/A"
        improvement = ((new_val - baseline_val) / baseline_val * 100).round(1)
        return f"{improvement:+.1f}%" if improvement else "0%"

    def get_stats_by_group(self, df: pd.DataFrame, group_by: str) -> pd.DataFrame:
        """Calculate statistics with round variance"""
        # First get mean per question/round
        question_means = df.groupby([group_by, "question"]).agg({
            "chunk_relevance": "mean",
            "answer_correctness": "mean",
            "code_reference": "mean",
            "weighted_total": "mean"
        })
        
        # Then get mean and std across questions
        return question_means.groupby(level=0).agg({
            "chunk_relevance": ["mean", "std"],
            "answer_correctness": ["mean", "std"],
            "code_reference": ["mean", "std"],
            "weighted_total": ["mean", "std"]
        }).round(2)

    def display_comparison_table(self, results: Dict[RetrievalMethod, pd.DataFrame]):
        """Display rich table comparing all methods"""
        table = Table(title="Method Comparison by Evaluator Model")
        
        # Add columns
        table.add_column("Metric", style="cyan")
        table.add_column("Model", style="magenta")
        for method in results.keys():
            table.add_column(method.value.replace("_", " ").title(), style="blue")
            if method != self.baseline_method:
                table.add_column(f"{method.value} Improvement", style="yellow")

        # Calculate stats for each method
        stats_by_method = {
            method: self.get_stats_by_group(df, "evaluator_model")
            for method, df in results.items()
        }

        metrics = ["chunk_relevance", "answer_correctness", "code_reference", "weighted_total"]
        baseline_stats = stats_by_method[self.baseline_method]

        for metric in metrics:
            for model in baseline_stats.index:
                row_data = [
                    metric.replace("_", " ").title(),
                    model.split(":")[-1]
                ]
                
                # Add data for each method
                for method in results.keys():
                    stats = stats_by_method[method]
                    mean = stats.loc[model, (metric, "mean")]
                    std = stats.loc[model, (metric, "std")]
                    row_data.append(f"{mean:.2f} ±{std:.2f}")
                    
                    # Add improvement column if not baseline
                    if method != self.baseline_method:
                        baseline_mean = baseline_stats.loc[model, (metric, "mean")]
                        row_data.append(self.calculate_improvement(mean, baseline_mean))

                table.add_row(*row_data)

        self.console.print(table)

    def display_environment_comparison(self, results: Dict[RetrievalMethod, pd.DataFrame]):
        """Display comparison across different evaluation environments"""
        table = Table(title="Method Comparison by Environment")
        
        table.add_column("Environment", style="cyan")
        for method in results.keys():
            table.add_column(method.value.replace("_", " ").title(), style="blue")
            if method != self.baseline_method:
                table.add_column(f"{method.value} Improvement", style="yellow")

        # Get environments from all results
        environments = sorted(set().union(*[
            set(df["environment"].unique()) 
            for df in results.values()
        ]))

        baseline_df = results[self.baseline_method]
        
        for env in environments:
            row_data = [env]
            
            for method in results.keys():
                df = results[method]
                env_score = df[df["environment"] == env]["weighted_total"].mean()
                row_data.append(f"{env_score:.2f}")
                
                if method != self.baseline_method:
                    baseline_score = baseline_df[
                        baseline_df["environment"] == env
                    ]["weighted_total"].mean()
                    row_data.append(self.calculate_improvement(env_score, baseline_score))
            
            table.add_row(*row_data)

        self.console.print(table)

    def analyze_results(self):
        """Analyze and display results comparison"""
        results = self.load_all_results()
        if not results:
            self.console.print("[red]No results found!")
            return

        # Display comparisons
        self.display_comparison_table(results)
        self.console.print("\n")
        self.display_environment_comparison(results)
        
        # Save detailed results
        self.save_detailed_results(results)

    def save_detailed_results(self, results: Dict[RetrievalMethod, pd.DataFrame]):
        """Save detailed results to CSV"""
        # Combine all results with method column
        dfs = []
        for method, df in results.items():
            df = df.copy()
            df["method"] = method.value
            dfs.append(df)
        
        combined_df = pd.concat(dfs)
        output_path = self.base_dir / "evaluation_comparison.csv"
        combined_df.to_csv(output_path, index=False)
        self.console.print(f"\nDetailed results saved to {output_path}")

if __name__ == "__main__":
    analyzer = ResultAnalyzer(
        Path("evaluations"), 
        baseline_method=RetrievalMethod.EMBEDDING
    )
    analyzer.analyze_results()