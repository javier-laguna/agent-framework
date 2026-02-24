"""Carga de configuración desde YAML y variables de entorno."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

# Ruta por defecto al YAML: configs/config.yaml (único archivo activo)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "config.yaml"


@dataclass
class LLMConfig:
    """Configuración del modelo de lenguaje."""

    model: str
    api_key: str
    provider: str = "openai"  # "openai" | "gemini"
    temperature: float = 0.0
    max_tokens: int = 1024


@dataclass
class ConversationConfig:
    """Configuración del historial de conversación."""

    enabled: bool = False
    max_messages: int = 10
    trim_when_over: int = 12
    summary_type: str = "facts"


def _load_yaml(path: Path) -> dict:
    """Carga el YAML y devuelve un diccionario; vacío si no existe."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_conversation_config(config_path: Path | None = None) -> ConversationConfig:
    """Carga la sección conversation del YAML.

    Args:
        config_path: Ruta al YAML. Si es None, usa configs/config.yaml.

    Returns:
        ConversationConfig. Si no existe la sección, enabled=False y valores por defecto.
    """
    path = config_path or _DEFAULT_CONFIG_PATH
    data = _load_yaml(path)
    conv = data.get("conversation") or {}
    return ConversationConfig(
        enabled=bool(conv.get("enabled", False)),
        max_messages=int(conv.get("max_messages", 10)),
        trim_when_over=int(conv.get("trim_when_over", 12)),
        summary_type=str(conv.get("summary_type", "facts")).lower(),
    )


def load_config(config_path: Path | None = None) -> LLMConfig:
    """Carga la configuración desde YAML y variables de entorno.

    El YAML define modelo, temperature y max_tokens bajo la clave 'llm'.
    OPENAI_API_KEY y OPENAI_MODEL (opcional) sobrescriben desde el entorno.

    Args:
        config_path: Ruta al archivo YAML. Si es None, usa configs/config.yaml.

    Returns:
        LLMConfig con los valores a usar por el adapter LLM.
    """
    path = config_path or _DEFAULT_CONFIG_PATH
    data = _load_yaml(path)
    provider = "openai"
    model = "gpt-4o-mini"
    temperature = 0.0
    max_tokens = 1024

    llm = data.get("llm") or {}
    provider = (llm.get("provider") or provider).lower()
    model = llm.get("model", model)
    temperature = float(llm.get("temperature", temperature))
    max_tokens = int(llm.get("max_tokens", max_tokens))

    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if os.environ.get("GEMINI_MODEL"):
            model = os.environ.get("GEMINI_MODEL", model)
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if os.environ.get("OPENAI_MODEL"):
            model = os.environ.get("OPENAI_MODEL", model)

    return LLMConfig(
        model=model,
        api_key=api_key,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
    )
