"""Tool: devuelve el detalle completo de una tabla (columnas, FKs, ejemplos)."""

from __future__ import annotations

import unicodedata
from pathlib import Path

from tools.registry import register_tool


def _get_schema_dir() -> Path:
    from core.config import load_supabase_config

    config = load_supabase_config()
    return Path(config.schema_dir).resolve()


def _normalize(name: str) -> str:
    """Normaliza nombre: lowercase, sin acentos, strip."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    return ascii_name.strip().lower()


@register_tool("schema_describe")
def schema_describe(tabla: str) -> str:
    """Devuelve la descripción completa de una tabla: columnas con tipos
    y comentarios, relaciones (foreign keys), notas de uso y ejemplos de
    consultas SQL. Úsala antes de escribir SQL contra una tabla específica
    para conocer su estructura exacta y los JOINs habituales.

    Args:
        tabla: Nombre de la tabla a describir (case-insensitive, sin acentos).
    """
    if not tabla or not tabla.strip():
        return "Debes indicar el nombre de la tabla. Llama primero a schema_list si no sabes cuáles existen."

    schema_dir = _get_schema_dir()
    if not schema_dir.exists():
        return (
            f"El directorio de schema no existe: {schema_dir}. "
            "Ejecuta `python scripts/generate_schema_docs.py` para inicializarlo."
        )

    target = _normalize(tabla)
    match: Path | None = None
    for f in schema_dir.glob("*.md"):
        if _normalize(f.stem) == target:
            match = f
            break

    if match is None:
        available = sorted(_normalize(f.stem) for f in schema_dir.glob("*.md"))
        if not available:
            return f"No hay tablas documentadas en {schema_dir}."
        sample = ", ".join(available[:20])
        more = f" (y {len(available) - 20} más)" if len(available) > 20 else ""
        return (
            f"No se encontró documentación para la tabla '{tabla}'. "
            f"Tablas disponibles: {sample}{more}."
        )

    try:
        return match.read_text(encoding="utf-8")
    except OSError as exc:
        return f"No se pudo leer la documentación de '{tabla}': {exc}"
