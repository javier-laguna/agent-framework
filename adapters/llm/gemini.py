"""Adaptador LLM para Google Gemini (vía DSPy/LiteLLM)."""

from dspy import LM

from adapters.llm.base import BaseLLMAdapter
from core.config import LLMConfig


class GeminiAdapter(BaseLLMAdapter):
    """Construye dspy.LM para modelos Gemini (Google AI Studio).

    Usa el prefijo gemini/ para la API con clave (no Vertex).
    Variable de entorno: GEMINI_API_KEY.
    """

    def get_lm(self, config: LLMConfig) -> LM:
        model_str = f"gemini/{config.model}"
        return LM(
            model_str,
            api_key=config.api_key or None,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
