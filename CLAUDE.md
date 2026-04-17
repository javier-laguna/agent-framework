# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run the Streamlit app** (primary entry point):
```bash
PYTHONPATH=. streamlit run scripts/streamlit_app.py
```

**Install dependencies** (Python 3.10 required):
```bash
# With uv (recommended)
uv venv && source .venv/bin/activate
uv pip install dspy streamlit chromadb pypdf python-docx python-dotenv

# Or with pipenv
pipenv install && pipenv shell
```

**Use programmatically**:
```python
from core.wrapper import DSPyWrapper
wrapper = DSPyWrapper()
response = wrapper.respond("mensaje")
```

## Configuration

Active config is always `configs/config.yaml`. Templates are in `config_examples/`. The YAML structure supports these top-level sections: `llm`, `conversation`, `agent`, `tools`, `vectorizer`.

API keys go in `.env` (never in YAML): `OPENAI_API_KEY` or `GEMINI_API_KEY`. Env vars `OPENAI_MODEL` / `GEMINI_MODEL` override the model in the YAML.

## Architecture

The system is a **DSPy-based LLM agent framework** with four concerns:

### 1. Core (`core/`)
- `config.py` — Loads YAML + env vars into typed dataclasses (`LLMConfig`, `ConversationConfig`, `VectorizerConfig`, `ToolsConfig`, `load_agent_description()`).
- `conversation.py` — In-memory conversation history with auto-trim and LLM-based fact summarization when `trim_when_over` threshold is exceeded.
- `wrapper.py` — `DSPyWrapper` is the main entry point. Uses `dspy.Predict` for plain Q&A or `dspy.ReAct` when tools are enabled. The DSPy signature switches between `"question -> answer"` and `"context, question -> answer"` depending on whether conversation memory is active.

### 2. LLM Adapters (`adapters/llm/`)
- `base.py` — `BaseLLMAdapter` ABC with `get_lm(config) -> dspy.LM`.
- `openai.py`, `gemini.py` — Concrete adapters. Selected by `config.provider` in `DSPyWrapper.__init__`.

### 3. RAG Pipeline (`rag/` + `adapters/vector/`)
- `rag/loader.py` — Loads PDF, DOCX, and plain text files into `Document` objects (with page map for PDFs).
- `rag/chunker.py` — Splits text into `Chunk` objects with character positions.
- `rag/ingest.py` — Orchestrates load → chunk → index. `ingest_bytes()` handles Streamlit file uploads (writes to temp file).
- `adapters/vector/chroma.py` — `ChromaAdapter` using local persistent ChromaDB. Always uses OpenAI embeddings (requires `OPENAI_API_KEY` even when using Gemini as the LLM).

### 4. Tools (`tools/`)
- `registry.py` — Central `_TOOL_REGISTRY` dict. Tools self-register via `@register_tool("name")` decorator.
- Tool modules (`datetime_now.py`, `rag_search.py`) are dynamically imported by name at startup when `tools.enabled = true` in config.
- To add a new tool: create `tools/my_tool.py`, use `@register_tool("my_tool")`, add `"my_tool"` to `tools.available` in the YAML.

### 5. UI (`scripts/streamlit_app.py`)
- Multi-tab Streamlit app. `DSPyWrapper` and `ChromaAdapter` are cached with `@st.cache_resource`.
- The app has tabs for: agent chat, document ingestion (RAG), and vector store management.
- Sidebar has a debug mode that shows backend state and full tracebacks.

## Key Design Decisions
- **No tests directory** — testing is done via the Streamlit UI or direct programmatic use.
- **Single active config** — `configs/config.yaml` is the only file read at runtime; `config_examples/` are not loaded automatically.
- **ChromaDB always requires OpenAI key** for embeddings, regardless of which LLM provider is selected.
- `PYTHONPATH=.` is required when running scripts, as all imports are absolute from the project root.
