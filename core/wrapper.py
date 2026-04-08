"""Wrapper sobre DSPy: configura el LM desde un adapter y expone respond(texto)."""

from pathlib import Path

import dspy

from adapters.llm.base import BaseLLMAdapter
from adapters.llm.gemini import GeminiAdapter
from adapters.llm.openai import OpenAIAdapter
from core.config import (
    load_agent_description,
    load_config,
    load_conversation_config,
    load_tools_config,
)
from core.conversation import ConversationMemory


class DSPyWrapper:
    """Inicializa DSPy con un adapter LLM y permite enviar texto y recibir respuesta."""

    def __init__(
        self,
        config_path: Path | None = None,
        adapter: BaseLLMAdapter | None = None,
        memory: ConversationMemory | None = None,
    ):
        path = config_path or None
        self._config = load_config(path)
        self._agent_description = load_agent_description(path)
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

        tools_config = load_tools_config(path)
        self._tools: list = []
        self._tools_enabled = False
        self._max_iters = tools_config.max_iters

        if tools_config.enabled and tools_config.available:
            self._tools = self._load_tools(tools_config.available)
            self._tools_enabled = bool(self._tools)

        has_context = False
        if memory is not None:
            self._memory: ConversationMemory | None = memory
            self._summary_type = _summary_type
            has_context = True
            self._predict_summary = self._make_summary_predictor(_summary_type)
        elif conv_config.enabled:
            self._memory = ConversationMemory(conv_config)
            self._summary_type = _summary_type
            has_context = True
            self._predict_summary = self._make_summary_predictor(_summary_type)
        else:
            self._memory = None
            self._summary_type = ""
            self._predict_summary = None

        sig_str = "context, question -> answer" if has_context else "question -> answer"
        self._predict = self._make_main_module(sig_str)

    def _load_tools(self, names: list[str]) -> list:
        """Importa los módulos de tools para que se registren y devuelve las funciones."""
        import importlib

        for name in names:
            try:
                importlib.import_module(f"tools.{name}")
            except ModuleNotFoundError:
                pass

        from tools.registry import get_tools
        return get_tools(names)

    def _make_main_module(self, signature_str: str):
        """Crea el módulo principal: ReAct si hay tools, Predict si no."""
        if self._agent_description:
            sig = dspy.Signature(signature_str, self._agent_description)
        else:
            sig = signature_str

        if self._tools_enabled:
            return dspy.ReAct(sig, tools=self._tools, max_iters=self._max_iters)
        return dspy.Predict(sig)

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
        """Genera resumen del bloque y devuelve un único facts_summary."""
        if not text.strip() or self._predict_summary is None:
            return ""
        pred = self._predict_summary(conversation_text=text)
        return getattr(pred, "facts_summary", "") or ""

    def respond(self, texto: str) -> str:
        """Envía el texto al modelo configurado y devuelve la respuesta.

        Si hay tools habilitadas, usa ReAct para que el modelo decida si las invoca.
        Si la memoria de conversación está habilitada, usa contexto previo.
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

    def has_tools(self) -> bool:
        """Indica si el wrapper tiene tools habilitadas."""
        return self._tools_enabled

    def get_tool_names(self) -> list[str]:
        """Devuelve los nombres de las tools activas."""
        return [fn.__name__ for fn in self._tools]

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
