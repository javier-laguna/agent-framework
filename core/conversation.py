"""Memoria de conversación: historial de mensajes y recorte con resumen/keywords."""

from collections.abc import Callable

from core.config import ConversationConfig


class ConversationMemory:
    """Mantiene mensajes recientes y un histórico (keywords o resumen) de lo recortado."""

    def __init__(self, config: ConversationConfig) -> None:
        self._config = config
        self._messages: list[dict[str, str]] = []  # {"role": "user"|"assistant", "content": str}
        self._historic: str = ""  # keywords o resumen de mensajes ya recortados

    def add_turn(self, user_content: str, assistant_content: str) -> None:
        """Añade un turno (mensaje usuario + respuesta asistente)."""
        self._messages.append({"role": "user", "content": user_content})
        self._messages.append({"role": "assistant", "content": assistant_content})

    def get_context(self) -> str:
        """Devuelve el contexto a inyectar en el prompt: histórico + últimos N mensajes formateados."""
        parts: list[str] = []
        if self._historic.strip():
            parts.append(f"[Resumen previo: {self._historic}]")
        n = self._config.max_messages
        recent = self._messages[-n:] if len(self._messages) > n else self._messages
        for m in recent:
            role = "Usuario" if m["role"] == "user" else "Asistente"
            parts.append(f"{role}: {m['content']}")
        return "\n\n".join(parts) if parts else ""

    def get_messages_for_display(self) -> list[dict[str, str]]:
        """Copia de los mensajes recientes para mostrar en UI (role, content)."""
        return [{"role": m["role"], "content": m["content"]} for m in self._messages]

    def get_historic(self) -> str:
        """Resumen/keywords de la parte ya recortada de la conversación."""
        return self._historic

    def trim_and_summarize(self, summarizer_fn: Callable[[str], str]) -> None:
        """Si hay más de trim_when_over mensajes, recorta y actualiza el histórico en un solo resumen."""
        over = self._config.trim_when_over
        keep = self._config.max_messages
        if len(self._messages) <= over:
            return
        to_summarize = self._messages[: -keep]
        self._messages = self._messages[-keep:]
        new_part = "\n\n".join(
            f"{'Usuario' if m['role'] == 'user' else 'Asistente'}: {m['content']}"
            for m in to_summarize
        )
        if self._historic.strip():
            full_text = f"[Resumen anterior]\n{self._historic}\n\n[Mensajes a incorporar]\n{new_part}"
        else:
            full_text = new_part
        self._historic = summarizer_fn(full_text)
