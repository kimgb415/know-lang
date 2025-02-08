from typing import List
from know_lang_bot.configs.config import ModelProvider, LLMConfig
from pydantic_ai.messages import ModelMessage

def _process_anthropic_batch(batched_input: List[List[ModelMessage]], config: LLMConfig) -> List[str]:
    """Helper function to process Anthropic LLM requests in batch."""
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    from pydantic_ai.models.anthropic import AnthropicAgentModel

    client = anthropic.Anthropic()
    requests : List[Request] = []
    for batch in batched_input:
        system_prompt, anthropic_prompt = AnthropicAgentModel._map_message(batch)
        requests.append(
            Request(
                custom_id="my-first-request",
                params=MessageCreateParamsNonStreaming(
                    model=config.model_name,
                    max_tokens=1024,
                    messages=system_prompt,
                    messages=anthropic_prompt,
                )
            )
        )

    message_batch = client.messages.batches.create(
        requests=requests,
    )

    print(message_batch)


def batch_process_requests(batched_input: List[List[ModelMessage]], config: LLMConfig) -> List[str]:
    if config.model_provider == ModelProvider.ANTHROPIC:
        return _process_anthropic_batch(batched_input, config)
    else:
        raise ValueError("Unsupported model provider for batch request processing")