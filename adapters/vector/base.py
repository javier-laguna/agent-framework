"""Interfaz base para adaptadores de almacén vectorial."""

from abc import ABC, abstractmethod


class BaseVectorAdapter(ABC):
    """Protocolo para adaptadores de vector store."""

    @abstractmethod
    def add_documents(self, texts: list[str], metadatas: list[dict] | None = None) -> int:
        """Indexa una lista de textos (chunks) en la colección.

        Args:
            texts: Fragmentos de texto a indexar.
            metadatas: Metadatos opcionales por cada fragmento.

        Returns:
            Cantidad de fragmentos indexados.
        """
        ...

    @abstractmethod
    def query(self, text: str, top_k: int = 5) -> list[dict]:
        """Busca los fragmentos más similares al texto dado.

        Args:
            text: Texto de consulta.
            top_k: Cantidad de resultados a devolver.

        Returns:
            Lista de dicts con claves 'text', 'metadata' y 'distance'.
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """Devuelve la cantidad de documentos indexados en la colección."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Elimina todos los documentos de la colección."""
        ...
