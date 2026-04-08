"""Orquestador de ingesta: carga, fragmenta e indexa documentos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from adapters.vector.base import BaseVectorAdapter
from core.config import VectorizerConfig
from rag.chunker import Chunk, chunk_text_with_positions
from rag.loader import Document, load_document


@dataclass
class IngestResult:
    """Resultado de la ingesta de un archivo."""

    filename: str
    chunks_indexed: int
    success: bool
    error: str = ""


def _resolve_pages(chunk: Chunk, page_map: list[dict]) -> list[int]:
    """Determina en qué página(s) cae un chunk dado su rango de caracteres."""
    pages: list[int] = []
    for pm in page_map:
        if chunk.start < pm["end"] and chunk.end > pm["start"]:
            pages.append(pm["page"])
    return pages or [1]


def _build_metadata(
    doc: Document, chunk: Chunk, chunk_index: int,
) -> dict:
    """Construye metadata para un chunk, incluyendo páginas si es PDF."""
    page_map = doc.metadata.get("page_map")
    meta = {
        "source": doc.metadata.get("source", ""),
        "type": doc.metadata.get("type", ""),
        "chunk_index": chunk_index,
    }
    if page_map:
        pages = _resolve_pages(chunk, page_map)
        meta["pages"] = ", ".join(str(p) for p in pages)
        meta["total_pages"] = doc.metadata.get("total_pages", 0)
    return meta


def ingest_file(
    file_path: Path,
    adapter: BaseVectorAdapter,
    config: VectorizerConfig,
    original_filename: str | None = None,
) -> IngestResult:
    """Carga un archivo, lo fragmenta e indexa en el vector store."""
    display_name = original_filename or file_path.name
    try:
        doc: Document = load_document(file_path)
        doc.metadata["source"] = display_name

        chunks = chunk_text_with_positions(
            doc.text,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        if not chunks:
            return IngestResult(
                filename=display_name, chunks_indexed=0, success=True,
            )

        texts = [c.text for c in chunks]
        metadatas = [_build_metadata(doc, c, i) for i, c in enumerate(chunks)]
        count = adapter.add_documents(texts=texts, metadatas=metadatas)
        return IngestResult(filename=display_name, chunks_indexed=count, success=True)

    except Exception as exc:
        return IngestResult(
            filename=display_name, chunks_indexed=0, success=False, error=str(exc),
        )


def ingest_bytes(
    file_bytes: bytes,
    filename: str,
    adapter: BaseVectorAdapter,
    config: VectorizerConfig,
) -> IngestResult:
    """Ingesta desde bytes en memoria (para archivos subidos via Streamlit)."""
    import tempfile

    suffix = Path(filename).suffix
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)
        return ingest_file(tmp_path, adapter, config, original_filename=filename)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
