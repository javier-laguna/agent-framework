"""Tool: devuelve la fecha y hora actual."""

import datetime

from tools.registry import register_tool


@register_tool("datetime_now")
def datetime_now() -> str:
    """Devuelve la fecha y hora actual del servidor en formato legible.
    Usa esta herramienta cuando el usuario pregunte por la fecha, hora o día actual.
    """
    now = datetime.datetime.now()
    return now.strftime("%A %d de %B de %Y, %H:%M:%S")
