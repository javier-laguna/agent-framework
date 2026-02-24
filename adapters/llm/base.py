"""Interfaz base para adaptadores de modelos de lenguaje."""

from abc import ABC, abstractmethod

from dspy import LM

from core.config import LLMConfig


class BaseLLMAdapter(ABC):
    """Protocolo para adaptadores que construyen un LM de DSPy a partir de configuración."""

    @abstractmethod
    def get_lm(self, config: LLMConfig) -> LM:
        """Construye y devuelve una instancia de dspy.LM configurada.

        No llama a dspy.configure(); eso lo hace el core.

        Args:
            config: Configuración (modelo, api_key, temperature, max_tokens).

        Returns:
            Instancia de dspy.LM lista para usar con dspy.configure(lm=...).
        """
        ...
