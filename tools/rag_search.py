"""Tool: busca información relevante en los documentos vectorizados."""

from __future__ import annotations

from tools.registry import register_tool

_adapter = None
_config = None


def _get_adapter():
    """Inicializa el adaptador vectorial de forma lazy (una sola vez)."""
    global _adapter, _config
    if _adapter is None:
        from adapters.vector.chroma import ChromaAdapter
        from core.config import load_vectorizer_config

        _config = load_vectorizer_config()
        if not _config.enabled:
            return None, _config
        _adapter = ChromaAdapter(_config)
    return _adapter, _config


@register_tool("rag_search")
def rag_search(query: str) -> str:
    """Busca informacion relevante en los documentos de la biblioteca del agente.
    Usa esta herramienta cuando el usuario haga preguntas que puedan responderse
    con los documentos cargados en la base de conocimiento. Devuelve fragmentos
    relevantes con su fuente para que puedas citar el origen de la informacion.
    """
    adapter, config = _get_adapter()
    if adapter is None:
        return "El vectorizador no esta habilitado en la configuracion."
    if adapter.count() == 0:
        return "No hay documentos indexados en la biblioteca."

    results = adapter.query(query, top_k=config.top_k)
    if not results:
        return "No se encontraron fragmentos relevantes para esta consulta."

    parts: list[str] = []
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        source = meta.get("source", "desconocido")
        chunk_idx = meta.get("chunk_index", "?")
        distance = r["distance"]
        text = r["text"]

        location = f"fuente: {source}, chunk: {chunk_idx}"
        pages = meta.get("pages")
        if pages:
            location += f", pagina(s): {pages}"
        total_pages = meta.get("total_pages")
        if total_pages:
            location += f" de {total_pages}"

        parts.append(
            f"[Fragmento {i}] ({location}, distancia: {distance:.3f})\n{text}"
        )

    return "\n\n---\n\n".join(parts)
