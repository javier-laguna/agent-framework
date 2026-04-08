# Framework DSPy

Wrapper sobre DSPy para desplegar agentes LLM con configuración por YAML y variables de entorno. Soporta OpenAI y Google Gemini. Incluye memoria de conversación con resumen actualizable, un sistema de herramientas (tools) extensible, un pipeline RAG con ChromaDB y una app Streamlit para interactuar con el agente.

## Funcionalidades

- **Wrapper DSPy**: Carga config (YAML + env), configura el LM y expone `respond(texto)`.
- **Adaptadores LLM**: OpenAI y Gemini (Google AI Studio). Elección por `llm.provider` en config.
- **Memoria de conversación**: Historial de mensajes; al superar un umbral se recorta y se genera un resumen que preserva datos concretos (nombre, edad, preferencias) y se actualiza en cada recorte en lugar de concatenar.
- **Sistema de tools**: Herramientas que el agente puede invocar via ReAct. Incluidas: `datetime_now` y `rag_search`. Se añaden nuevas con el decorador `@register_tool`.
- **Pipeline RAG**: Ingesta de documentos PDF, DOCX y texto plano; fragmentación con overlap; indexación en ChromaDB local. El agente recupera fragmentos relevantes antes de responder.
- **App Streamlit**: Dos pestañas — agente conversacional y gestión de documentos RAG. Modo debug en sidebar.

## Instalación

Requisito: Python 3.10.

Con **uv** (recomendado):

```bash
cd /ruta/al/proyecto
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install dspy streamlit chromadb pypdf python-docx python-dotenv
```

O con **Pipfile** (pipenv):

```bash
pipenv install
pipenv shell
```

### API key en entorno

Crea un archivo `.env` en la raíz del proyecto (no se sube a git):

**OpenAI:**

```
OPENAI_API_KEY=sk-...
```

**Gemini (Google AI Studio):**

```
GEMINI_API_KEY=...
```

> **Nota:** ChromaDB siempre usa embeddings de OpenAI (`text-embedding-3-small` por defecto), por lo que `OPENAI_API_KEY` es necesaria incluso si el LLM es Gemini.

Opcional: `OPENAI_MODEL` o `GEMINI_MODEL` para sobrescribir el modelo del YAML.

## Configuración

El archivo activo es `configs/config.yaml`. Los archivos en `config_examples/` son plantillas; copia el que quieras a `configs/config.yaml`.

### Ejemplo completo (OpenAI + RAG + tools)

```yaml
llm:
  provider: openai
  model: gpt-4o-mini
  temperature: 0.0
  max_tokens: 1024

agent:
  description: >
    Eres un experto en seguridad vial. Respondes en español.
    Si usas la herramienta de búsqueda, cita la fuente con número de página.

conversation:
  enabled: true
  max_messages: 12
  trim_when_over: 13
  summary_type: facts

tools:
  enabled: true
  max_iters: 5
  available:
    - datetime_now
    - rag_search

vectorizer:
  enabled: true
  embedding_model: text-embedding-3-small
  collection_name: documents
  persist_directory: ./data/chroma
  chunk_size: 1000
  chunk_overlap: 150
  top_k: 10
```

### Parámetros

| Sección       | Parámetro          | Descripción |
|---------------|--------------------|-------------|
| `llm`         | `provider`         | `openai` o `gemini`. |
| `llm`         | `model`            | Nombre del modelo (ej. `gpt-4o-mini`, `gemini-2.0-flash`). |
| `llm`         | `temperature`      | 0.0 a 1.0. |
| `llm`         | `max_tokens`       | Máximo de tokens en la respuesta. |
| `agent`       | `description`      | System prompt del agente (texto libre). |
| `conversation`| `enabled`          | `true` para usar historial y resumen. |
| `conversation`| `max_messages`     | Número de mensajes recientes a mantener. |
| `conversation`| `trim_when_over`   | A partir de cuántos mensajes se recorta y actualiza el resumen. |
| `conversation`| `summary_type`     | Solo `facts` (preserva datos concretos para consultas posteriores). |
| `tools`       | `enabled`          | `true` para activar el modo ReAct con herramientas. |
| `tools`       | `available`        | Lista de nombres de tools a cargar (ej. `datetime_now`, `rag_search`). |
| `tools`       | `max_iters`        | Máximo de iteraciones ReAct por llamada. |
| `vectorizer`  | `enabled`          | `true` para activar ChromaDB y RAG. |
| `vectorizer`  | `embedding_model`  | Modelo de embeddings OpenAI (ej. `text-embedding-3-small`). |
| `vectorizer`  | `collection_name`  | Nombre de la colección en ChromaDB. |
| `vectorizer`  | `persist_directory`| Ruta local donde ChromaDB guarda los datos. |
| `vectorizer`  | `chunk_size`       | Tamaño máximo de cada fragmento en caracteres. |
| `vectorizer`  | `chunk_overlap`    | Solapamiento entre fragmentos en caracteres. |
| `vectorizer`  | `top_k`            | Número de fragmentos a recuperar por consulta. |

La API key se toma del entorno (`.env` o variables de sistema) según `provider`; no se pone en el YAML.

## Ejecución

Desde la raíz del proyecto, con el entorno activado:

```bash
PYTHONPATH=. streamlit run scripts/streamlit_app.py
```

Abre la URL que muestre la terminal (p. ej. http://localhost:8501).

La app tiene dos pestañas:
- **Agente**: envía mensajes al modelo, ve el historial y el resumen del historial recortado. Muestra las tools activas si las hay.
- **RAG - Documentos**: sube documentos (PDF, DOCX, TXT), vectorízalos, gestiona la biblioteca y realiza búsquedas semánticas directas.

En la barra lateral puedes activar **Modo debug** para ver el estado del backend y tracebacks completos si hay error.

Logs más verbosos de Streamlit en terminal:

```bash
PYTHONPATH=. streamlit run scripts/streamlit_app.py --logger.level=debug
```

## Añadir una tool nueva

1. Crea `tools/mi_tool.py` con el decorador `@register_tool`:

```python
from tools.registry import register_tool

@register_tool("mi_tool")
def mi_tool(argumento: str) -> str:
    """Descripción que verá el LLM al decidir si usar esta herramienta."""
    return resultado
```

2. Agrega `"mi_tool"` a `tools.available` en `configs/config.yaml`.

## Uso programático

```python
from pathlib import Path
from core.wrapper import DSPyWrapper

wrapper = DSPyWrapper()  # usa configs/config.yaml y .env
respuesta = wrapper.respond("¿Qué es DSPy?")
print(respuesta)
```

Opcional: `DSPyWrapper(config_path=Path("otro.yaml"))` para otro archivo de config.

## Estructura del proyecto

```
.
├── .env                    # No versionado; crear a mano con API keys
├── .gitignore
├── Pipfile
├── README.md
├── configs/
│   └── config.yaml         # Config activa (único YAML en uso)
├── config_examples/
│   ├── openai.yaml
│   └── gemini.yaml
├── core/
│   ├── config.py           # Carga YAML y env (LLM, conversation, tools, vectorizer)
│   ├── conversation.py     # Memoria: historial y resumen
│   └── wrapper.py          # DSPyWrapper, respond()
├── adapters/
│   ├── llm/
│   │   ├── base.py         # BaseLLMAdapter
│   │   ├── openai.py
│   │   └── gemini.py
│   └── vector/
│       ├── base.py         # BaseVectorAdapter
│       └── chroma.py       # ChromaDB con persistencia local
├── rag/
│   ├── loader.py           # Carga PDF, DOCX, TXT → Document
│   ├── chunker.py          # Fragmentación con posiciones de caracteres
│   └── ingest.py           # Orquestador: carga → fragmenta → indexa
├── tools/
│   ├── registry.py         # Registro central de tools (@register_tool)
│   ├── datetime_now.py
│   └── rag_search.py
└── scripts/
    └── streamlit_app.py
```
