"""Adaptador de base de datos para Supabase (Postgres) con conexión psycopg.

Provee tres garantías de solo lectura, en orden de defensa:
1. (Externa, configurada por el usuario) Rol de Postgres con permisos
   restringidos en el DSN de SUPABASE_DB_URL.
2. Transacción BEGIN READ ONLY: Postgres rechaza cualquier write a nivel de
   sesión aunque el SQL pase la validación.
3. Validación léxica del SQL: rechaza statements que no sean SELECT/WITH y
   detecta stacked queries antes de tocar la BD (mejor mensaje al LLM).
"""

from __future__ import annotations

import re
import time

import sqlparse
from psycopg_pool import ConnectionPool

from adapters.database.base import BaseDBAdapter
from core.config import SupabaseConfig


_FORBIDDEN_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "ALTER",
    "CREATE", "GRANT", "REVOKE", "COPY", "CALL", "DO", "MERGE",
    "REINDEX", "VACUUM", "CLUSTER", "LOCK", "REFRESH",
}

_LIMIT_REGEX = re.compile(r"\blimit\s+\d+", re.IGNORECASE)


class ReadOnlyViolation(ValueError):
    """SQL rechazado por la validación de solo lectura."""


class SupabaseAdapter(BaseDBAdapter):
    """Adaptador Postgres con pool de conexiones y enforcement de solo lectura."""

    def __init__(self, config: SupabaseConfig) -> None:
        if not config.db_url:
            raise RuntimeError(
                "SUPABASE_DB_URL no está configurado. Define la variable de "
                "entorno con el connection string de Postgres."
            )
        self._config = config
        self._pool: ConnectionPool | None = None

    def _get_pool(self) -> ConnectionPool:
        if self._pool is None:
            self._pool = ConnectionPool(
                conninfo=self._config.db_url,
                min_size=self._config.pool_min,
                max_size=self._config.pool_max,
                open=True,
                kwargs={"application_name": "servicio_social_agent"},
            )
        return self._pool

    @staticmethod
    def _validate_read_only(sql: str) -> None:
        """Rechaza SQL que no sea SELECT/WITH o que tenga múltiples statements."""
        clean = sql.strip().rstrip(";").strip()
        if not clean:
            raise ReadOnlyViolation("La consulta está vacía.")

        statements = [s for s in sqlparse.parse(sql) if str(s).strip().rstrip(";").strip()]
        if len(statements) > 1:
            raise ReadOnlyViolation(
                "Solo se permite un statement por llamada (stacked queries bloqueadas)."
            )

        stmt = statements[0]
        first_keyword = None
        for token in stmt.flatten():
            if token.ttype in (sqlparse.tokens.Keyword, sqlparse.tokens.Keyword.DML, sqlparse.tokens.Keyword.DDL, sqlparse.tokens.Keyword.CTE):
                first_keyword = token.normalized.upper()
                break
        if first_keyword not in {"SELECT", "WITH"}:
            raise ReadOnlyViolation(
                f"Solo se permiten SELECT/WITH. Detectado: '{first_keyword or 'desconocido'}'."
            )

        for token in stmt.flatten():
            if token.ttype in (sqlparse.tokens.Keyword, sqlparse.tokens.Keyword.DML, sqlparse.tokens.Keyword.DDL, sqlparse.tokens.Keyword.CTE):
                kw = token.normalized.upper()
                if kw in _FORBIDDEN_KEYWORDS:
                    raise ReadOnlyViolation(
                        f"Palabra clave prohibida en consulta de solo lectura: '{kw}'."
                    )

    def _apply_implicit_limit(self, sql: str) -> tuple[str, bool]:
        """Si la query no trae LIMIT, envuelve en subquery con LIMIT implícito.

        Devuelve (sql_modificado, se_aplicó_limit).
        """
        if _LIMIT_REGEX.search(sql):
            return sql, False
        clean = sql.strip().rstrip(";").strip()
        wrapped = f"SELECT * FROM ({clean}) AS _agent_limited LIMIT {self._config.implicit_row_limit}"
        return wrapped, True

    def query(self, sql: str, params: tuple | None = None) -> dict:
        self._validate_read_only(sql)
        final_sql, limit_applied = self._apply_implicit_limit(sql)

        start = time.perf_counter()
        pool = self._get_pool()
        with pool.connection() as conn:
            conn.autocommit = False
            with conn.cursor() as cur:
                cur.execute("BEGIN READ ONLY")
                cur.execute(f"SET LOCAL statement_timeout = {self._config.statement_timeout_ms}")
                try:
                    cur.execute(final_sql, params)
                    columns = [d.name for d in cur.description] if cur.description else []
                    rows_raw = cur.fetchall() if cur.description else []
                finally:
                    conn.rollback()  # cierra la transacción sin commit (no había writes igual)

        rows = [list(r) for r in rows_raw]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        truncated = limit_applied and len(rows) >= self._config.implicit_row_limit
        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "truncated": truncated,
            "elapsed_ms": elapsed_ms,
            "sql_executed": final_sql,
            "sql_original": sql,
        }

    def list_tables(self) -> list[str]:
        result = self.query(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' "
            "ORDER BY table_name"
        )
        return [r[0] for r in result["rows"]]

    def describe_table(self, name: str) -> dict:
        cols_result = self.query(
            """
            SELECT
                c.column_name,
                c.data_type,
                c.is_nullable,
                c.column_default,
                pgd.description
            FROM information_schema.columns c
            LEFT JOIN pg_catalog.pg_statio_all_tables st
                ON c.table_schema = st.schemaname AND c.table_name = st.relname
            LEFT JOIN pg_catalog.pg_description pgd
                ON pgd.objoid = st.relid AND pgd.objsubid = c.ordinal_position
            WHERE c.table_schema = 'public' AND c.table_name = %s
            ORDER BY c.ordinal_position
            """,
            (name,),
        )
        columns = [
            {
                "name": r[0],
                "type": r[1],
                "nullable": r[2] == "YES",
                "default": r[3],
                "comment": r[4],
            }
            for r in cols_result["rows"]
        ]

        fks_result = self.query(
            """
            SELECT
                kcu.column_name,
                ccu.table_name AS references_table,
                ccu.column_name AS references_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage ccu
                ON ccu.constraint_name = tc.constraint_name
                AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_schema = 'public'
                AND tc.table_name = %s
            ORDER BY kcu.ordinal_position
            """,
            (name,),
        )
        foreign_keys = [
            {"column": r[0], "references_table": r[1], "references_column": r[2]}
            for r in fks_result["rows"]
        ]

        refs_result = self.query(
            """
            SELECT
                tc.table_name AS from_table,
                kcu.column_name AS from_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage ccu
                ON ccu.constraint_name = tc.constraint_name
                AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_schema = 'public'
                AND ccu.table_name = %s
            ORDER BY tc.table_name, kcu.ordinal_position
            """,
            (name,),
        )
        referenced_by = [
            {"table": r[0], "column": r[1]}
            for r in refs_result["rows"]
        ]

        return {
            "name": name,
            "columns": columns,
            "foreign_keys": foreign_keys,
            "referenced_by": referenced_by,
        }

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None
