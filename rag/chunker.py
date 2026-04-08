"""Fragmentación de texto en chunks respetando límites de oración."""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+")


@dataclass
class Chunk:
    """Fragmento de texto con su posición en el texto original."""

    text: str
    start: int
    end: int


def _split_sentences(text: str) -> list[tuple[str, int]]:
    """Divide texto en oraciones, devolviendo (texto, posicion_inicio)."""
    results: list[tuple[str, int]] = []
    for match in re.finditer(r"\S.*?(?:[.!?;]\s|$)", text, re.DOTALL):
        s = match.group().strip()
        if s:
            results.append((s, match.start()))
    if not results and text.strip():
        results.append((text.strip(), 0))
    return results


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Divide un texto en fragmentos respetando límites de oración.

    Wrapper retrocompatible que devuelve solo los textos.
    """
    return [c.text for c in chunk_text_with_positions(text, chunk_size, chunk_overlap)]


def chunk_text_with_positions(
    text: str, chunk_size: int = 500, chunk_overlap: int = 50,
) -> list[Chunk]:
    """Divide un texto en fragmentos con posiciones en el texto original.

    Args:
        text: Texto completo a fragmentar.
        chunk_size: Cantidad máxima de caracteres por fragmento.
        chunk_overlap: Caracteres aproximados de solapamiento entre fragmentos.

    Returns:
        Lista de Chunk con texto y posiciones start/end.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    if len(text) <= chunk_size:
        return [Chunk(text=text, start=0, end=len(text))]

    sentences = _split_sentences(text)
    if not sentences:
        return [Chunk(text=text, start=0, end=len(text))]

    chunks: list[Chunk] = []
    current_sentences: list[tuple[str, int]] = []
    current_len = 0

    for sent_text, sent_start in sentences:
        added_len = len(sent_text) + (1 if current_sentences else 0)

        if current_len + added_len > chunk_size and current_sentences:
            chunk_str = " ".join(s for s, _ in current_sentences)
            chunk_start = current_sentences[0][1]
            chunk_end = current_sentences[-1][1] + len(current_sentences[-1][0])
            chunks.append(Chunk(text=chunk_str, start=chunk_start, end=chunk_end))

            overlap_sentences: list[tuple[str, int]] = []
            overlap_len = 0
            for item in reversed(current_sentences):
                if overlap_len + len(item[0]) > chunk_overlap and overlap_sentences:
                    break
                overlap_sentences.insert(0, item)
                overlap_len += len(item[0]) + 1

            current_sentences = overlap_sentences
            current_len = sum(len(s) for s, _ in current_sentences) + max(0, len(current_sentences) - 1)

        current_sentences.append((sent_text, sent_start))
        current_len += added_len

    if current_sentences:
        chunk_str = " ".join(s for s, _ in current_sentences)
        chunk_start = current_sentences[0][1]
        chunk_end = current_sentences[-1][1] + len(current_sentences[-1][0])
        if chunks and chunk_str == chunks[-1].text:
            pass
        else:
            chunks.append(Chunk(text=chunk_str, start=chunk_start, end=chunk_end))

    return chunks
