from pydantic_ai.models import Model, KnownModelName
from know_lang_bot.core.types import ModelProvider
from typing import get_args

def create_pydantic_model(
    model_provider: ModelProvider,
    model_name: str,
) -> Model | KnownModelName:
    model_str = f"{model_provider}:{model_name}"

    if model_str in get_args(KnownModelName):
        return model_str
    elif model_provider == ModelProvider.HUGGINGFACE:
        from know_lang_bot.models.huggingface import HuggingFaceModel
        return HuggingFaceModel(model_name=model_name)
    elif model_provider == ModelProvider.TESTING:
        # should be used for testing purposes only
        pass
    else:
        raise NotImplementedError(f"Model {model_provider}:{model_name} is not supported")
