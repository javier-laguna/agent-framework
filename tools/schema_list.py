"""Tool: lista las tablas documentadas en schema/tablas/ con su descripción."""

from __future__ import annotations

from pathlib import Path

from tools.registry import register_tool


def _parse_frontmatter(text: str) -> dict:
    """Extrae el frontmatter YAML simple (tabla, descripcion) sin dep externa."""
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    block = text[3:end].strip()
    fm: dict[str, str] = {}
    for line in block.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            fm[key.strip()] = val.strip()
    return fm


def _get_schema_dir() -> Path:
    from core.config import load_supabase_config

    config = load_supabase_config()
    return Path(config.schema_dir).resolve()


@register_tool("schema_list")
def schema_list() -> str:
    """Lista las tablas disponibles en la base de datos con su descripción
    de una línea. Úsala primero, antes de construir cualquier consulta SQL,
    para descubrir qué tablas existen y para qué sirven.

    Si necesitas detalle de una tabla específica (columnas, tipos, relaciones,
    ejemplos), llama a schema_describe(nombre_tabla).
    """
    schema_dir = _get_schema_dir()
    if not schema_dir.exists():
        return (
            f"El directorio de schema no existe: {schema_dir}. "
            "Ejecuta `python scripts/generate_schema_docs.py` para inicializarlo."
        )

    md_files = sorted(schema_dir.glob("*.md"))
    if not md_files:
        return (
            f"No hay tablas documentadas en {schema_dir}. "
            "Ejecuta `python scripts/generate_schema_docs.py` para generar plantillas."
        )

    lines = ["Tablas disponibles:"]
    for f in md_files:
        try:
            text = f.read_text(encoding="utf-8")
        except OSError:
            continue
        fm = _parse_frontmatter(text)
        nombre = fm.get("tabla") or f.stem
        descripcion = fm.get("descripcion") or "(sin descripción)"
        lines.append(f"- {nombre}: {descripcion}")
    return "\n".join(lines)
