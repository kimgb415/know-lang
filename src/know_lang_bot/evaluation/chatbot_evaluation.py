from typing import List, Dict, Optional
from enum import Enum
from pydantic import BaseModel, Field, computed_field
from pydantic_ai import Agent
from know_lang_bot.config import AppConfig
from know_lang_bot.utils.model_provider import create_pydantic_model
from know_lang_bot.chat_bot.chat_graph import ChatResult, process_chat
import asyncio
import datetime
from pathlib import Path

class EvalMetric(str, Enum):
    CHUNK_RELEVANCE = "chunk_relevance"
    ANSWER_CORRECTNESS = "answer_correctness"
    CODE_REFERENCE = "code_reference"

class EvalCase(BaseModel):
    """Single evaluation case focused on code understanding"""
    question: str
    expected_files: List[str] = Field(description="Files that should be in retrieved chunks")
    expected_concepts: List[str] = Field(description="Key concepts that should be in answer")
    expected_code_refs: List[str] = Field(description="Code references that should be mentioned")
    difficulty: int = Field(ge=1, le=3, description="1: Easy, 2: Medium, 3: Hard")


class MetricScores(BaseModel):
    chunk_relevance: float = Field(ge=0.0, le=10.0, description="Score for chunk relevance")
    answer_correctness: float = Field(ge=0.0, le=10.0, description="Score for answer correctness")
    code_reference: float = Field(ge=0.0, le=10.0, description="Score for code reference quality")

    @computed_field
    def weighted_total(self) -> float:
        """Calculate weighted total score"""
        weights = {
            "chunk_relevance": 0.4,
            "answer_correctness": 0.4,
            "code_reference": 0.2
        }
        return sum(
            getattr(self, metric) * weight 
            for metric, weight in weights.items()
        )

class EvalAgentResponse(MetricScores):
    """Raw response from evaluation agent"""
    feedback: str

class EvalResult(BaseModel):
    """Evaluation result with scores and feedback"""
    evaluator_model: str
    case: EvalCase
    eval_response: EvalAgentResponse

class ChatBotEvaluationContext(EvalCase, ChatResult):
    pass

class EvalSummary(EvalResult, ChatResult):
    """Evaluation summary with chat and evaluation results"""
    pass

    
class ChatBotEvaluator:
    def __init__(self, config: AppConfig):
        """Initialize evaluator with app config"""
        self.config = config
        self.eval_agent = Agent(
            create_pydantic_model(
                model_provider=config.evaluator.model_provider,
                model_name=config.evaluator.model_name
            ),
            system_prompt=self._build_eval_prompt(),
            result_type=EvalAgentResponse
        )

    def _build_eval_prompt(self) -> str:
        return """You are an expert evaluator of code understanding systems.
Evaluate the response based on these specific criteria:

1. Chunk Relevance (0-1):
- Are the retrieved code chunks from the expected files?
- Do they contain relevant code sections?

2. Answer Correctness (0-1):
- Does the answer accurately explain the code?
- Are the expected concepts covered?

3. Code Reference Quality (0-1):
- Does it properly cite specific code locations?
- Are code references clear and relevant?

Format your response as JSON:
{
    "chunk_relevance": float type score (from 0.0f to 10.0f),
    "answer_correctness": float type score (from 0.0f to 10.0f),
    "code_reference": float type score (from 0.0f to 10.0f),
    "feedback": "Brief explanation of scores"
}
"""

    async def evaluate_single(
        self,
        case: EvalCase,
        chat_result: ChatResult
    ) -> EvalResult:
        """Evaluate a single case"""
        # Prepare evaluation context
        eval_context = ChatBotEvaluationContext(
            **case.model_dump(),
            **chat_result.model_dump()
        )

        # Get evaluation from the model
        result = await self.eval_agent.run(
            eval_context.model_dump_json(),
        )
        eval_response : EvalAgentResponse = result.data

        return EvalResult(
            case=case,
            eval_response=eval_response,
            evaluator_model=f"{self.config.evaluator.model_provider}:{self.config.evaluator.model_name}"
        )

    async def evaluate_batch(
        self,
        cases: List[EvalCase],
        process_chat_func,
        max_concurrent: int = 2
    ) -> List[EvalResult]:
        """Run evaluation on multiple cases with concurrency control"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def eval_single_with_limit(case: EvalCase) -> EvalResult:
            async with semaphore:
                chat_result = await process_chat_func(case.question)
                return await self.evaluate_single(case, chat_result)

        return await asyncio.gather(
            *[eval_single_with_limit(case) for case in cases]
        )

# src/transformers/quantizers/base.py
TRANSFORMER_QUANTIZER_BASE_CASES = [
    EvalCase(
        question= "How are different quantization methods implemented in the transformers library, and what are the key components required to implement a new quantization method?",
        expected_files= ["quantizers/base.py"],
        expected_concepts= [
            "HfQuantizer abstract base class",
            "PreTrainedModel quantization",
            "pre/post processing of models",
            "quantization configuration", 
            "requires_calibration flag"
        ],
        expected_code_refs= [
            "class HfQuantizer",
            "preprocess_model method",
            "postprocess_model method",
            "_process_model_before_weight_loading",
            "requires_calibration attribute"
        ],
        difficulty= 3
    )
]

# src/transformers/quantizers/auto.py
TRANSFORMER_QUANTIZER_AUTO_CASES = [
    EvalCase(
        question="How does the transformers library automatically select and configure the appropriate quantization method, and what happens when loading a pre-quantized model?",
        expected_files=[
            "quantizers/auto.py",
            "utils/quantization_config.py"
        ],
        expected_concepts=[
            "automatic quantizer selection",
            "quantization config mapping",
            "config merging behavior",
            "backwards compatibility for bitsandbytes",
            "quantization method resolution"
        ],
        expected_code_refs=[
            "AUTO_QUANTIZER_MAPPING",
            "AUTO_QUANTIZATION_CONFIG_MAPPING",
            "AutoHfQuantizer.from_config",
            "AutoQuantizationConfig.from_pretrained",
            "merge_quantization_configs method"
        ],
        difficulty=3
    )
]


# src/transformers/pipelines/base.py
TRANSFORMER_PIPELINE_BASE_TEST_CASES = [
    EvalCase(
        question="How does the Pipeline class handle model and device initialization?",
        expected_files=["base.py"],
        expected_concepts=[
            "device placement",
            "model initialization",
            "framework detection",
            "device type detection",
            "torch dtype handling"
        ],
        expected_code_refs=[
            "def __init__",
            "def device_placement",
            "infer_framework_load_model",
            "self.device = torch.device"
        ],
        difficulty=3
    ),
    EvalCase(
        question="How does the Pipeline class implement batched inference and data loading?",
        expected_files=["base.py", "pt_utils.py"],
        expected_concepts=[
            "batch processing",
            "data loading",
            "collate function",
            "padding implementation",
            "iterator pattern"
        ],
        expected_code_refs=[
            "def get_iterator",
            "class PipelineDataset",
            "class PipelineIterator",
            "_pad",
            "pad_collate_fn"
        ],
        difficulty=3
    )
]

# src/transformers/pipelines/text_generation.py
TRANSFORMER_PIPELINE_TEXT_GENERATION_TEST_CASES = [
    EvalCase(
        question="How does the TextGenerationPipeline handle chat-based generation and template processing?",
        expected_files=["text_generation.py", "base.py"],
        expected_concepts=[
            "chat message formatting",
            "template application",
            "message continuation",
            "role handling",
            "assistant prefill behavior"
        ],
        expected_code_refs=[
            "class Chat",
            "tokenizer.apply_chat_template",
            "continue_final_message",
            "isinstance(prompt_text, Chat)",
            "postprocess"
        ],
        difficulty=3
    )
]

# src/transformers/generation/logits_process.py
TRANSFORMER_LOGITS_PROCESSOR_TEST_CASES = [
    EvalCase(
        question="How does TopKLogitsWarper implement top-k filtering for text generation?",
        expected_files=["generation/logits_process.py"],
        expected_concepts=[
            "top-k filtering algorithm",
            "probability masking",
            "batch processing",
            "logits manipulation",
            "vocabulary filtering"
        ],
        expected_code_refs=[
            "class TopKLogitsWarper(LogitsProcessor)",
            "torch.topk(scores, top_k)[0]",
            "indices_to_remove = scores < torch.topk",
            "scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)",
            "top_k = max(top_k, min_tokens_to_keep)"
        ],
        difficulty=3
    ),
    EvalCase(
        question="How does TemperatureLogitsProcessor implement temperature sampling for controlling generation randomness?",
        expected_files=["generation/logits_process.py"],
        expected_concepts=[
            "temperature scaling",
            "probability distribution shaping",
            "logits normalization",
            "generation randomness control",
            "batch processing with temperature"
        ],
        expected_code_refs=[
            "class TemperatureLogitsProcessor(LogitsProcessor)",
            "scores_processed = scores / self.temperature",
            "if not isinstance(temperature, float) or not (temperature > 0)",
            "def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor)",
            "raise ValueError(except_msg)"
        ],
        difficulty=3
    )
]

# src/transformers/trainer.py
TRANSFORMER_TRAINER_TEST_CASES = [
    EvalCase(
        question="How does Trainer handle distributed training and gradient accumulation? Explain the implementation details.",
        expected_files=["trainer.py"],
        expected_concepts=[
            "gradient accumulation steps",
            "distributed training logic",
            "optimizer step scheduling",
            "loss scaling",
            "device synchronization"
        ],
        expected_code_refs=[
            "def training_step",
            "def _wrap_model",
            "self.accelerator.backward",
            "self.args.gradient_accumulation_steps",
            "if args.n_gpu > 1",
            "model.zero_grad()"
        ],
        difficulty=3
    ),
    EvalCase(
        question="How does the Trainer class implement custom optimizer and learning rate scheduler creation? Explain the initialization process and supported configurations.",
        expected_files=["trainer.py"],
        expected_concepts=[
            "optimizer initialization",
            "learning rate scheduler",
            "weight decay handling",
            "optimizer parameter groups",
            "AdamW configuration",
            "custom optimizer support"
        ],
        expected_code_refs=[
            "def create_optimizer",
            "def create_scheduler",
            "get_decay_parameter_names",
            "optimizer_grouped_parameters",
            "self.args.learning_rate",
            "optimizer_kwargs"
        ],
        difficulty=3
    )
]

TRANSFORMER_TEST_CASES : List[EvalCase] = [
    *TRANSFORMER_QUANTIZER_BASE_CASES,
    *TRANSFORMER_QUANTIZER_AUTO_CASES,
    *TRANSFORMER_PIPELINE_BASE_TEST_CASES,
    *TRANSFORMER_PIPELINE_TEXT_GENERATION_TEST_CASES,
    *TRANSFORMER_LOGITS_PROCESSOR_TEST_CASES,
    *TRANSFORMER_TRAINER_TEST_CASES,
]


async def main():
    from rich.console import Console
    from rich.pretty import Pretty
    import json
    import chromadb
    console = Console()
    config = AppConfig()
    evaluator = ChatBotEvaluator(config)
    collection = chromadb.PersistentClient(path=str(config.db.persist_directory)).get_collection(name=config.db.collection_name)

    summary_list : List[EvalSummary] = []

    for case in TRANSFORMER_TEST_CASES:
        try:
            chat_result : ChatResult = await process_chat(question=case.question, collection=collection, config=config)
            result : EvalResult = await evaluator.evaluate_single(case, chat_result)
            
            eval_summary = EvalSummary(
                **chat_result.model_dump(),
                **result.model_dump()
            )
            summary_list.append(eval_summary)

            import time
            time.sleep(5) # Sleep for 5 seconds to avoid rate limiting

        except Exception:
            console.print_exception()
    
    # Write the final JSON array to a file
    
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    file_name = Path("evaluations", f"transformers_{config.evaluator.model_provider}_evaluation_results_{current_date}.json")
    with open(file_name, "w") as f:
        json_list = [summary.model_dump() for summary in summary_list]
        json.dump(json_list, f, indent=2)


    console.print(Pretty(summary_list))

if __name__ == "__main__":
    asyncio.run(main())