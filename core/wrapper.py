"""Wrapper sobre DSPy: configura el LM desde un adapter y expone respond(texto)."""

from pathlib import Path

import dspy

from adapters.llm.base import BaseLLMAdapter
from adapters.llm.gemini import GeminiAdapter
from adapters.llm.openai import OpenAIAdapter
from core.config import load_config, load_conversation_config
from core.conversation import ConversationMemory


class DSPyWrapper:
    """Inicializa DSPy con un adapter LLM y permite enviar texto y recibir respuesta."""

    def __init__(
        self,
        config_path: Path | None = None,
        adapter: BaseLLMAdapter | None = None,
        memory: ConversationMemory | None = None,
    ):
        """Carga config, obtiene el LM del adapter y configura DSPy.

        Args:
            config_path: Ruta al YAML. Si es None, usa configs/config.yaml.
            adapter: Adaptador que construye el LM. Si es None, se elige por config.provider (openai | gemini).
            memory: Memoria de conversación opcional. Si es None y conversation.enabled en config, se crea una internamente.
        """
        path = config_path or None
        self._config = load_config(path)
        if adapter is not None:
            self._adapter = adapter
        elif self._config.provider == "gemini":
            self._adapter = GeminiAdapter()
        else:
            self._adapter = OpenAIAdapter()
        lm = self._adapter.get_lm(self._config)
        dspy.configure(lm=lm)

        conv_config = load_conversation_config(path)
        _summary_type = (conv_config.summary_type or "facts").lower()
        if memory is not None:
            self._memory: ConversationMemory | None = memory
            self._summary_type = _summary_type
            self._predict = dspy.Predict("context, question -> answer")
            self._predict_summary = self._make_summary_predictor(_summary_type)
        elif conv_config.enabled:
            self._memory = ConversationMemory(conv_config)
            self._summary_type = _summary_type
            self._predict = dspy.Predict("context, question -> answer")
            self._predict_summary = self._make_summary_predictor(_summary_type)
        else:
            self._memory = None
            self._summary_type = ""
            self._predict = dspy.Predict("question -> answer")
            self._predict_summary = None

    def _make_summary_predictor(self, summary_type: str) -> dspy.Predict:
        """Crea el predictor de resumen (solo tipo facts)."""
        facts_instruction = (
            "Recibes un texto que puede incluir un [Resumen anterior] y [Mensajes a incorporar]. "
            "Actualiza y fusiona todo en un único resumen breve que preserve los datos concretos del usuario: "
            "nombre, edad, números, preferencias, hechos. Sin repetir; un solo párrafo. "
            "Objetivo: poder responder después preguntas como '¿cómo se llama?' o '¿cuántos años tiene?'. "
            "Escribe solo el resumen (ej: 'Usuario se llama X, tiene Y años, le gusta Z')."
        )
        facts_sig = dspy.Signature("conversation_text -> facts_summary", facts_instruction)
        return dspy.Predict(facts_sig)

    def _summarize_chunk(self, text: str) -> str:
        """Genera resumen del bloque (incluye resumen n-1 si existe) y devuelve un único facts_summary."""
        if not text.strip() or self._predict_summary is None:
            return ""
        pred = self._predict_summary(conversation_text=text)
        return getattr(pred, "facts_summary", "") or ""

    def respond(self, texto: str) -> str:
        """Envía el texto al modelo configurado y devuelve la respuesta.

        Si la memoria de conversación está habilitada, usa contexto previo y actualiza el historial.

        Args:
            texto: Entrada (pregunta o mensaje) a enviar al LLM.

        Returns:
            Respuesta generada por el modelo.
        """
        if self._memory is None:
            pred = self._predict(question=texto)
            return getattr(pred, "answer", "")

        context = self._memory.get_context()
        pred = self._predict(context=context or "(sin contexto previo)", question=texto)
        answer = getattr(pred, "answer", "")
        self._memory.add_turn(texto, answer)
        self._memory.trim_and_summarize(self._summarize_chunk)
        return answer

    def has_conversation(self) -> bool:
        """Indica si el wrapper usa memoria de conversación."""
        return self._memory is not None

    def get_conversation_messages(self) -> list[dict[str, str]]:
        """Mensajes recientes para mostrar en UI (vacío si no hay memoria)."""
        if self._memory is None:
            return []
        return self._memory.get_messages_for_display()

    def get_historic_summary(self) -> str:
        """Resumen/keywords del historial ya recortado (vacío si no hay memoria)."""
        if self._memory is None:
            return ""
        return self._memory.get_historic()
