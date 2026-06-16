---
name: agregar-tool
description: "Agrega una nueva tool al agente DSPy de este proyecto (tools/ + registry + config.yaml). Usar cuando el usuario pida 'agrega una tool de X', 'crea una herramienta para X', 'quiero que el agente pueda consultar/calcular X', o 'add a tool for X'."
---

# Skill: agregar-tool

Agrega una tool nueva al agente DSPy de este repo siguiendo los patrones existentes. Una tool es una función en `tools/<nombre>.py` decorada con `@register_tool("<nombre>")`, registrada en `tools.available` de `configs/config.yaml`. **Convención cuádruple obligatoria**: nombre de módulo == nombre de función == nombre registrado == entrada YAML. `core/wrapper.py` y `tools/registry.py` nunca se tocan.

## Fuente de verdad

1. **Antes de generar nada**, lee `docs/agregar_tools.md`, sección "Parte 2: recetario para agentes" — ahí están los invariantes de naming (2.1), la tabla archivo-por-archivo (2.2), las plantillas P1–P5 (2.3), los comandos de verificación (2.4) y los criterios de documentación (2.5–2.6).
2. **Lee SIEMPRE además una tool real** del tipo que vas a crear, como referencia viva:
   - T1 (determinística pura) → `tools/datetime_now.py`
   - T2 (config YAML / catálogo en disco) → `tools/schema_list.py`
   - T3 (API externa + secreto en `.env`) → ejemplo `weather_now` en la sección 1.11 de la guía
   - T4 (adapter / recurso costoso) → `tools/rag_search.py` + `adapters/vector/chroma.py`
   - T5 (side-effects para la UI) → `tools/supabase_query.py` + `scripts/streamlit_app.py`
3. **Si la guía contradice al código real, el código gana.** Avisa al usuario del drift detectado y ofrece actualizar la guía. Si la guía no existe, deriva el patrón directamente de la tool de referencia.

## Workflow

1. **Clasifica** la tool pedida con el árbol de decisión (guía 1.4) en T1–T5 (pueden combinarse; clasifica por el rasgo dominante). Anuncia al usuario el tipo elegido y qué archivos va a implicar (tabla 2.2).
2. **Recopila lo que falte** preguntando al usuario solo lo no inferible: nombre en `snake_case`, argumentos y tipos, proveedor/endpoint si es API, el **NOMBRE** de la variable de entorno (nunca pidas ni escribas el VALOR del secreto), parámetros tunables para el YAML, y si quiere render especial en la UI.
3. **Genera `tools/<nombre>.py`** desde la plantilla correspondiente (P1–P5, guía 2.3), respetando el contrato: docstring en español orientado al LLM, firma tipada simple, retorno siempre `str`, toda excepción atrapada y devuelta como mensaje, output acotado.
4. **Registra** la tool en `configs/config.yaml` → `tools.available`.
5. **Condicionales** según el tipo:
   - Sección nueva en el YAML + dataclass `<Nombre>Config` y `load_<nombre>_config()` en `core/config.py` (default `enabled: bool = False`).
   - Adapter: `adapters/<dominio>/base.py` (ABC) + implementación, con lazy-init en la tool.
   - Secreto: agrega `<ENV_VAR>=` (línea sin valor) a `.env.example` **y** a `.env`; el usuario pega el valor.
   - Dependencia nueva: declárala en `Pipfile` e instálala con `VIRTUAL_ENV=.venv uv pip install <paquete>` (este proyecto usa uv, no pip).
   - Si agregaste sección al YAML: sincroniza los templates de `config_examples/`.
6. **UI (solo T5, opcional)**: integra en `scripts/streamlit_app.py` con el patrón clear-antes / get-después / `_render_*` (guía P5). Recuerda que la trayectoria ReAct ya muestra cualquier tool sin esto.
7. **Verifica** con los comandos exactos de la guía 2.4, en orden (registro → función con camino feliz y de error → `DSPyWrapper().get_tool_names()`). Si algo falla, corrige antes de continuar. Ojo con la trampa dotenv: `python -c` no carga `.env` solo.
8. **Documenta**: bullet en "### 4. Tools (`tools/`)" de `CLAUDE.md` (+ env var en "## Configuration", + dependencia en "## Commands" si aplica). Toca `agent.description` del YAML **solo** si el agente necesita protocolo de uso (criterio guía 2.5).
9. **Reporta** con el checklist de la guía 2.6: archivos creados/modificados, pendientes manuales (valor del secreto en `.env`), salida de la verificación, y una pregunta de prueba para Streamlit que fuerce el uso de la tool.

## Reglas duras (no negociables)

- Secretos jamás en YAML, en código ni en mensajes de error — solo nombres de variables; los valores viven en `.env`, que está gitignoreado.
- El retorno de la tool es siempre `str`; jamás propagar excepciones (ReAct debe poder leer el error y reintentar).
- Input de la tool = input hostil (lo escribe el LLM): validar siempre; nunca `eval`/`exec`/shell/SQL concatenado.
- Output acotado: límites de filas/caracteres con nota de truncado; nunca JSON crudo de APIs.
- Timeout en todo I/O (red, base de datos).
- Docstrings, comentarios y mensajes en español.
