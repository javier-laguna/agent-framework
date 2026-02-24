"""Adaptador LLM para OpenAI (vía DSPy)."""

from dspy import LM

from adapters.llm.base import BaseLLMAdapter
from core.config import LLMConfig


class OpenAIAdapter(BaseLLMAdapter):
    """Construye dspy.LM para modelos OpenAI."""

    def get_lm(self, config: LLMConfig) -> LM:
        model_str = f"openai/{config.model}"
        return LM(
            model_str,
            api_key=config.api_key or None,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
