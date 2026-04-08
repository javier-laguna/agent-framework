"""Adaptador vectorial para ChromaDB."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from adapters.vector.base import BaseVectorAdapter
from core.config import VectorizerConfig


class ChromaAdapter(BaseVectorAdapter):
    """Implementación de vector store usando ChromaDB con persistencia local."""

    def __init__(self, config: VectorizerConfig, api_key: str = "") -> None:
        persist_path = Path(config.persist_directory).resolve()
        persist_path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(persist_path))
        self._ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            model_name=config.embedding_model,
        )
        self._collection = self._client.get_or_create_collection(
            name=config.collection_name,
            embedding_function=self._ef,
        )

    def add_documents(self, texts: list[str], metadatas: list[dict] | None = None) -> int:
        if not texts:
            return 0
        ids = [uuid.uuid4().hex for _ in texts]
        metas = metadatas or [{} for _ in texts]
        self._collection.add(documents=texts, metadatas=metas, ids=ids)
        return len(texts)

    def query(self, text: str, top_k: int = 5) -> list[dict]:
        results = self._collection.query(query_texts=[text], n_results=top_k)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        return [
            {"text": d, "metadata": m, "distance": dist}
            for d, m, dist in zip(docs, metas, dists)
        ]

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        name = self._collection.name
        ef = self._ef
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name, embedding_function=ef,
        )
