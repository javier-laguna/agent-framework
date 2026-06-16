"""Genera plantillas Markdown en schema/tablas/ a partir de information_schema.

Uso:
    PYTHONPATH=. python scripts/generate_schema_docs.py            # todas
    PYTHONPATH=. python scripts/generate_schema_docs.py usuarios   # solo una
    PYTHONPATH=. python scripts/generate_schema_docs.py --force    # sobrescribe

Por defecto NO sobrescribe archivos existentes (preserva edits manuales).
Genera frontmatter con `descripcion: TODO - describir esta tabla` para que
quede claro qué hay que rellenar a mano después.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from adapters.database.supabase import SupabaseAdapter
from core.config import load_supabase_config


def _render_table_md(meta: dict) -> str:
    """Construye el contenido Markdown de una tabla a partir de su metadata."""
    name = meta["name"]
    columns = meta["columns"]
    fks = meta["foreign_keys"]
    refs = meta["referenced_by"]

    lines: list[str] = []
    lines.append("---")
    lines.append(f"tabla: {name}")
    lines.append("descripcion: TODO - describir esta tabla en una línea")
    lines.append("---")
    lines.append("")
    lines.append(f"# {name}")
    lines.append("")
    lines.append("TODO: descripción extendida (qué representa, ciclo de vida, reglas de negocio).")
    lines.append("")
    lines.append("## Columnas")
    lines.append("")
    lines.append("| Columna | Tipo | Nullable | Default | Descripción |")
    lines.append("|---------|------|----------|---------|-------------|")
    for c in columns:
        col_name = c["name"]
        col_type = c["type"]
        nullable = "sí" if c["nullable"] else "no"
        default = c.get("default") or ""
        if default:
            default = str(default).replace("|", "\\|")
            if len(default) > 40:
                default = default[:37] + "..."
        comment = c.get("comment") or ""
        if comment:
            comment = str(comment).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {col_name} | {col_type} | {nullable} | {default} | {comment} |")
    lines.append("")
    lines.append("## Relaciones")
    lines.append("")
    if fks:
        for fk in fks:
            lines.append(
                f"- `{fk['column']}` → `{fk['references_table']}.{fk['references_column']}` (muchos-a-uno)"
            )
    else:
        lines.append("- (sin foreign keys salientes)")
    if refs:
        ref_strs = [f"`{r['table']}.{r['column']}`" for r in refs]
        lines.append(f"- Referenciada por: {', '.join(ref_strs)}.")
    lines.append("")
    lines.append("## Notas de uso")
    lines.append("")
    lines.append("TODO: convenciones, filtros frecuentes, gotchas.")
    lines.append("")
    lines.append("## Ejemplos")
    lines.append("")
    lines.append("```sql")
    lines.append(f"-- TODO: agregar ejemplos de queries útiles para {name}")
    lines.append(f"SELECT * FROM {name} LIMIT 10;")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tabla", nargs="?", help="Nombre de tabla específica (opcional)")
    parser.add_argument("--force", action="store_true", help="Sobrescribe archivos existentes")
    args = parser.parse_args()

    config = load_supabase_config()
    if not config.enabled:
        print("ERROR: supabase.enabled = false en configs/config.yaml.", file=sys.stderr)
        return 1
    if not config.db_url:
        print("ERROR: SUPABASE_DB_URL no está configurado en el entorno.", file=sys.stderr)
        return 1

    schema_dir = Path(config.schema_dir).resolve()
    schema_dir.mkdir(parents=True, exist_ok=True)

    adapter = SupabaseAdapter(config)
    try:
        if args.tabla:
            tables = [args.tabla]
        else:
            tables = adapter.list_tables()
        print(f"Detectadas {len(tables)} tabla(s): {', '.join(tables) if len(tables) <= 20 else f'{len(tables)} tablas'}")

        nuevas = 0
        saltadas = 0
        sobrescritas = 0
        errores: list[tuple[str, str]] = []

        for t in tables:
            dest = schema_dir / f"{t}.md"
            if dest.exists() and not args.force:
                saltadas += 1
                continue
            try:
                meta = adapter.describe_table(t)
                if not meta["columns"]:
                    errores.append((t, "no se encontraron columnas (¿existe en public?)"))
                    continue
                content = _render_table_md(meta)
                existed = dest.exists()
                dest.write_text(content, encoding="utf-8")
                if existed:
                    sobrescritas += 1
                else:
                    nuevas += 1
                print(f"  {'[force] ' if existed else ''}escrito: {dest.relative_to(_ROOT)}")
            except Exception as exc:
                errores.append((t, str(exc)))

        print()
        print(f"Resumen: {nuevas} nueva(s), {sobrescritas} sobrescrita(s), {saltadas} saltada(s) (ya existían).")
        if errores:
            print(f"Errores en {len(errores)} tabla(s):")
            for t, err in errores:
                print(f"  - {t}: {err}")
            return 2
        return 0
    finally:
        adapter.close()


if __name__ == "__main__":
    sys.exit(main())
