# Contexto de schema para el agente

Este directorio contiene la documentación de cada tabla expuesta a la herramienta
`supabase_query`. Es la **fuente de verdad** sobre qué ve el LLM: solo las tablas
con un `.md` aquí aparecen en `schema_list`, independientemente de lo que exista
en la base de datos.

## Estructura

```
schema/
├── README.md          # este archivo
└── tablas/
    └── <tabla>.md     # un archivo por tabla, en snake_case
```

## Convenciones de naming

- Un archivo por tabla, con el nombre exacto de la tabla en Postgres.
- Nombre del archivo en `snake_case`, lowercase, sin acentos.
- Extensión `.md`.

## Formato de cada archivo

```markdown
---
tabla: nombre_tabla
descripcion: Descripción de una línea (la usa schema_list).
---

# nombre_tabla

Descripción extendida en prosa: qué representa la tabla, ciclo de vida de
sus filas, reglas de negocio importantes.

## Columnas

| Columna   | Tipo        | Nullable | Descripción                |
|-----------|-------------|----------|----------------------------|
| id        | bigint      | no       | PK auto-incremental        |
| ...       | ...         | ...      | ...                        |

## Relaciones

- `col_x` → `otra_tabla.id` (muchos-a-uno): explicación corta.
- Referenciada por: `tabla_a.fk`, `tabla_b.fk`.

## Notas de uso

- Convenciones de soft-delete, filtros frecuentes, gotchas conocidos.

## Ejemplos

\`\`\`sql
-- Patrón común de JOIN o query útil
SELECT ...
FROM nombre_tabla
JOIN otra_tabla ON ...;
\`\`\`
```

## Cómo generar plantillas iniciales

```bash
PYTHONPATH=. python scripts/generate_schema_docs.py            # todas las tablas
PYTHONPATH=. python scripts/generate_schema_docs.py usuarios   # solo una
PYTHONPATH=. python scripts/generate_schema_docs.py --force    # sobrescribe
```

El script lee `information_schema` de Supabase y crea un `.md` por tabla con
columnas y FKs detectados. Después de generarlo, **edita a mano** la descripción,
notas de uso y ejemplos: esa es la parte de mayor valor para el agente y no se
puede inferir del schema solo.

## Qué pasa cuando la BD cambia

Los `.md` no se sincronizan automáticamente. Si agregas/quitas/renombras columnas
en Supabase, regenera la plantilla afectada con `--force` y vuelve a rellenar
las partes editadas a mano.
