# Framework DSPy

Wrapper sobre DSPy para desplegar agentes LLM con configuración por YAML y variables de entorno. Soporta OpenAI y Google Gemini. Incluye memoria de conversación con resumen actualizable y una app Streamlit para probar el backend.

## Funcionalidades

- **Wrapper DSPy**: Carga config (YAML + env), configura el LM y expone `respond(texto)`.
- **Adaptadores LLM**: OpenAI y Gemini (Google AI Studio). Elección por `llm.provider` en config.
- **Memoria de conversación**: Historial de mensajes; al superar un umbral se recorta y se genera un resumen que preserva datos concretos (nombre, edad, preferencias) y se actualiza en cada recorte en lugar de concatenar.
- **App Streamlit**: Interfaz para enviar mensajes, ver respuesta, historial y resumen del historial recortado. Modo debug en sidebar. Cuadro de texto se limpia tras enviar.

## Instalación

Requisito: Python 3.10.

Con **uv** (recomendado):

```bash
cd /ruta/al/proyecto
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install dspy streamlit
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

Opcional: `OPENAI_MODEL` o `GEMINI_MODEL` para sobrescribir el modelo del YAML.

## Configuración

El archivo activo es `configs/config.yaml`. Los archivos en `config_examples/` son plantillas; copia el que quieras a `configs/config.yaml`.

### Ejemplo mínimo (OpenAI)

```yaml
llm:
  provider: openai
  model: gpt-4o-mini
  temperature: 0.0
  max_tokens: 1024

conversation:
  enabled: false
  max_messages: 10
  trim_when_over: 12
  summary_type: facts
```

### Ejemplo Gemini

```yaml
llm:
  provider: gemini
  model: gemini-2.0-flash
  temperature: 0.0
  max_tokens: 1024

conversation:
  enabled: true
  max_messages: 10
  trim_when_over: 12
  summary_type: facts
```

### Parámetros

| Sección       | Parámetro      | Descripción |
|---------------|----------------|-------------|
| `llm`         | `provider`     | `openai` o `gemini`. |
| `llm`         | `model`        | Nombre del modelo (ej. `gpt-4o-mini`, `gemini-2.0-flash`). |
| `llm`         | `temperature`  | 0.0 a 1.0. |
| `llm`         | `max_tokens`   | Máximo de tokens en la respuesta. |
| `conversation`| `enabled`      | `true` para usar historial y resumen. |
| `conversation`| `max_messages` | Número de mensajes recientes a mantener. |
| `conversation`| `trim_when_over` | A partir de cuántos mensajes se recorta y se actualiza el resumen. |
| `conversation`| `summary_type` | Solo `facts` (resumen que preserva datos para preguntas posteriores). |

La API key se toma del entorno (`.env` o variables de sistema) según `provider`; no se pone en el YAML.

## Ejecución

Desde la raíz del proyecto, con el entorno activado:

```bash
PYTHONPATH=. streamlit run scripts/streamlit_app.py
```

Abre la URL que muestre la terminal (p. ej. http://localhost:8501). En la barra lateral puedes activar "Modo debug" para ver el estado del backend y tracebacks completos si hay error.

Logs más verbosos de Streamlit en terminal:

```bash
PYTHONPATH=. streamlit run scripts/streamlit_app.py --logger.level=debug
```

## Estructura del proyecto

```
.
├── .env                 # No versionado; crear a mano con API keys
├── .gitignore
├── Pipfile
├── README.md
├── configs/
│   └── config.yaml      # Config activa (único YAML en uso)
├── config_examples/
│   ├── openai.yaml
│   └── gemini.yaml
├── core/
│   ├── __init__.py
│   ├── config.py        # Carga YAML y env (LLM, conversation)
│   ├── conversation.py  # Memoria: historial y resumen
│   └── wrapper.py       # DSPyWrapper, respond()
├── adapters/
│   └── llm/
│       ├── __init__.py
│       ├── base.py      # BaseLLMAdapter
│       ├── openai.py
│       └── gemini.py
└── scripts/
    ├── __init__.py
    └── streamlit_app.py
```

## Uso programático

```python
from pathlib import Path
from core.wrapper import DSPyWrapper

wrapper = DSPyWrapper()  # usa configs/config.yaml y .env
respuesta = wrapper.respond("¿Qué es DSPy?")
print(respuesta)
```

Opcional: `DSPyWrapper(config_path=Path("otro.yaml"))` para otro archivo de config.
