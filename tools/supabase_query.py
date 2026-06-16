"""Tool: ejecuta una consulta SQL de solo lectura contra Supabase (Postgres).

Mantiene un buffer module-level con la última ronda de queries ejecutadas
para que la UI de Streamlit pueda mostrarlas debajo de la respuesta.
"""

from __future__ import annotations

from tools.registry import register_tool

_adapter = None
_config = None

# Buffer de queries ejecutadas durante la respuesta actual.
# Streamlit lo limpia antes de cada wrapper.respond() y lo lee al terminar.
_LAST_QUERIES: list[dict] = []


def _get_adapter():
    """Inicialización lazy del adaptador Supabase (una sola vez)."""
    global _adapter, _config
    if _adapter is None:
        from adapters.database.supabase import SupabaseAdapter
        from core.config import load_supabase_config

        _config = load_supabase_config()
        if not _config.enabled:
            return None, _config
        _adapter = SupabaseAdapter(_config)
    return _adapter, _config


def get_last_queries() -> list[dict]:
    """Devuelve una copia de las queries ejecutadas en la ronda actual."""
    return list(_LAST_QUERIES)


def clear_last_queries() -> None:
    """Vacía el buffer (la UI debe llamarlo antes de cada respond())."""
    _LAST_QUERIES.clear()


def _record_query(result: dict, error: str | None = None) -> None:
    """Appendea la query ejecutada al buffer para que la UI la muestre."""
    _LAST_QUERIES.append({
        "sql_original": result.get("sql_original") if result else None,
        "sql_executed": result.get("sql_executed") if result else None,
        "columns": result.get("columns", []) if result else [],
        "rows": result.get("rows", []) if result else [],
        "row_count": result.get("row_count", 0) if result else 0,
        "truncated": result.get("truncated", False) if result else False,
        "elapsed_ms": result.get("elapsed_ms", 0.0) if result else 0.0,
        "error": error,
    })


def _format_as_markdown_table(columns: list[str], rows: list[list]) -> str:
    """Formatea filas como tabla Markdown para que el LLM la lea/cite."""
    if not columns:
        return "(sin columnas)"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body_lines = []
    for row in rows:
        cells = []
        for v in row:
            if v is None:
                cells.append("NULL")
            else:
                s = str(v).replace("|", "\\|").replace("\n", " ")
                if len(s) > 200:
                    s = s[:197] + "..."
                cells.append(s)
        body_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *body_lines])


@register_tool("supabase_query")
def supabase_query(sql: str) -> str:
    """Ejecuta una consulta SQL de solo lectura contra la base de datos
    Supabase (Postgres). Solo se permiten SELECT y WITH; INSERT/UPDATE/DELETE
    y cualquier DDL serán rechazados.

    Antes de usar esta herramienta, llama a schema_list para conocer las
    tablas disponibles y a schema_describe para ver columnas, tipos y
    relaciones de cada tabla. Aprovecha las relaciones (FKs) para construir
    JOINs cuando la respuesta requiera datos de varias tablas.

    Devuelve los resultados como tabla en formato Markdown. Si la consulta
    no incluye LIMIT, se aplica automáticamente un LIMIT para evitar
    desbordar el contexto; en ese caso la respuesta lo indica.
    """
    adapter, config = _get_adapter()
    if adapter is None:
        msg = "La integración con Supabase no está habilitada en la configuración."
        _record_query({"sql_original": sql, "sql_executed": None}, error=msg)
        return msg

    try:
        result = adapter.query(sql)
    except Exception as exc:
        err_msg = f"Error al ejecutar la consulta: {exc}"
        _record_query({"sql_original": sql, "sql_executed": None}, error=err_msg)
        return err_msg

    _record_query(result)

    if result["row_count"] == 0:
        note = ""
        if result["truncated"]:
            note = f"\n\n(Se aplicó LIMIT {config.implicit_row_limit} automático.)"
        return f"La consulta se ejecutó correctamente pero no devolvió filas.{note}"

    table = _format_as_markdown_table(result["columns"], result["rows"])
    suffix = f"\n\n{result['row_count']} fila(s), {result['elapsed_ms']:.0f} ms."
    if result["truncated"]:
        suffix += (
            f" Resultados truncados al LIMIT implícito ({config.implicit_row_limit}). "
            "Refina la consulta o agrega un LIMIT explícito si necesitas un rango distinto."
        )
    return table + suffix
