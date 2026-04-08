"""Registro central de tools disponibles para el agente."""

from __future__ import annotations

from typing import Callable

_TOOL_REGISTRY: dict[str, Callable] = {}


def register_tool(name: str):
    """Decorador para registrar una función como tool del agente.

    Uso:
        @register_tool("mi_tool")
        def mi_tool(arg: str) -> str:
            '''Descripción que verá el LLM.'''
            return resultado
    """
    def decorator(fn: Callable) -> Callable:
        _TOOL_REGISTRY[name] = fn
        return fn
    return decorator


def get_tools(names: list[str]) -> list[Callable]:
    """Devuelve la lista de funciones tool correspondientes a los nombres dados.

    Ignora nombres que no existen en el registro (con warning en stderr).
    """
    import sys

    tools = []
    for name in names:
        fn = _TOOL_REGISTRY.get(name)
        if fn is None:
            print(f"[tools/registry] WARNING: tool '{name}' no encontrada en el registro.", file=sys.stderr)
            continue
        tools.append(fn)
    return tools


def list_available() -> list[str]:
    """Devuelve los nombres de todas las tools registradas."""
    return sorted(_TOOL_REGISTRY.keys())
