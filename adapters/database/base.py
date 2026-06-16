"""Interfaz base para adaptadores de base de datos relacional."""

from abc import ABC, abstractmethod


class BaseDBAdapter(ABC):
    """Protocolo para adaptadores de base de datos SQL."""

    @abstractmethod
    def query(self, sql: str, params: tuple | None = None) -> dict:
        """Ejecuta una consulta SQL de solo lectura.

        Args:
            sql: Consulta SQL (debe iniciar con SELECT o WITH).
            params: Parámetros opcionales para binding seguro.

        Returns:
            Dict con claves 'columns' (list[str]), 'rows' (list[list]),
            'row_count' (int), 'truncated' (bool), 'elapsed_ms' (float).
        """
        ...

    @abstractmethod
    def list_tables(self) -> list[str]:
        """Devuelve los nombres de tablas disponibles en el schema público."""
        ...

    @abstractmethod
    def describe_table(self, name: str) -> dict:
        """Devuelve metadatos de una tabla.

        Args:
            name: Nombre de la tabla.

        Returns:
            Dict con claves 'columns' (list de dicts con name, type, nullable,
            default, comment), 'foreign_keys' (list de dicts con column,
            references_table, references_column), 'referenced_by' (list de
            dicts con table, column).
        """
        ...
