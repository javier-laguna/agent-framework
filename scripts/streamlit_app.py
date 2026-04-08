"""App Streamlit para probar el wrapper DSPy: enviar texto y ver la respuesta.

Para ver más logs de Streamlit en la terminal, ejecuta:
  streamlit run scripts/streamlit_app.py --logger.level=debug
"""

import sys
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

import streamlit as st

from core.config import (
    load_agent_description,
    load_config,
    load_conversation_config,
    load_vectorizer_config,
)
from core.wrapper import DSPyWrapper


@st.cache_resource
def get_wrapper() -> DSPyWrapper:
    """Crea el wrapper una sola vez y lo cachea."""
    return DSPyWrapper()


@st.cache_resource
def get_vector_adapter():
    """Crea el adaptador vectorial una sola vez (None si no está habilitado)."""
    from adapters.vector.chroma import ChromaAdapter

    vec_config = load_vectorizer_config()
    if not vec_config.enabled:
        return None, vec_config
    adapter = ChromaAdapter(vec_config)
    return adapter, vec_config


def tab_agente(config, conv_config, agent_description, api_key_ok, debug):
    """Pestaña principal del agente conversacional."""
    st.header("Envía un mensaje al modelo")
    st.caption(
        f"Config: {config.provider} | API key {'configurada' if api_key_ok else 'no configurada'} | Modelo: {config.model}"
    )
    if agent_description:
        with st.expander("Descripción del agente (system prompt)", expanded=False):
            st.markdown(agent_description)
    if conv_config.enabled:
        st.caption("Modo conversación: el modelo usa el historial reciente.")

    try:
        wrapper = get_wrapper()
    except Exception as e:
        st.error(f"Error al crear el wrapper: {e}")
        if debug:
            st.sidebar.code(traceback.format_exc())
        return

    if debug:
        st.sidebar.caption(f"Wrapper listo. Conversación: {wrapper.has_conversation()}")

    if wrapper.has_tools():
        tool_names = ", ".join(wrapper.get_tool_names())
        st.caption(f"Tools activas: {tool_names}")

    if wrapper.has_conversation():
        messages = wrapper.get_conversation_messages()
        historic = wrapper.get_historic_summary()
        if historic:
            with st.expander("Resumen del historial recortado"):
                st.caption(historic)
        if messages:
            with st.expander("Historial de la conversación", expanded=False):
                for m in messages:
                    label = "Usuario" if m["role"] == "user" else "Asistente"
                    st.text(f"{label}: {m['content']}")

    if "last_response" not in st.session_state:
        st.session_state.last_response = None

    mensaje = st.text_area(
        "Mensaje",
        placeholder="Escribe tu pregunta o mensaje...",
        height=120,
        key="mensaje_draft",
    )

    if st.button("Obtener respuesta"):
        if not mensaje or not mensaje.strip():
            st.warning("Escribe un mensaje antes de enviar.")
            return
        if not api_key_ok:
            st.error("Configura OPENAI_API_KEY o GEMINI_API_KEY en el entorno o en un archivo .env.")
            return
        with st.spinner("Generando respuesta..."):
            try:
                if debug:
                    st.sidebar.caption("Llamando a wrapper.respond()...")
                respuesta = wrapper.respond(mensaje.strip())
                st.session_state.last_response = respuesta
                st.session_state.clear_input_next = True
                if debug:
                    st.sidebar.caption("Respond OK.")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
                st.code(traceback.format_exc(), language="text")

    if st.session_state.last_response is not None:
        st.divider()
        st.subheader("Respuesta")
        st.write(st.session_state.last_response)


def _get_docs_dir() -> Path:
    """Devuelve la ruta al directorio de documentos persistentes."""
    docs_dir = _ROOT / "rag_docs"
    docs_dir.mkdir(exist_ok=True)
    return docs_dir


def _save_uploaded_file(file_bytes: bytes, filename: str) -> Path:
    """Guarda un archivo subido en docs/ y devuelve la ruta."""
    dest = _get_docs_dir() / filename
    dest.write_bytes(file_bytes)
    return dest


def _list_library() -> list[dict]:
    """Lista los archivos almacenados en docs/ con metadata basica."""
    import datetime

    docs_dir = _get_docs_dir()
    files = []
    for f in sorted(docs_dir.iterdir()):
        if f.is_file() and not f.name.startswith("."):
            stat = f.stat()
            files.append({
                "nombre": f.name,
                "extension": f.suffix.lower(),
                "tamano_kb": round(stat.st_size / 1024, 1),
                "fecha": datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                "ruta": f,
            })
    return files


def tab_rag(debug):
    """Pestaña de gestión de documentos y búsqueda RAG."""
    from rag.ingest import ingest_bytes
    from rag.loader import SUPPORTED_EXTENSIONS

    st.header("RAG - Documentos")

    adapter, vec_config = get_vector_adapter()

    if adapter is None:
        st.warning("El vectorizador no está habilitado. Activa `vectorizer.enabled: true` en config.yaml.")
        return

    col_info, col_actions = st.columns([3, 1])
    with col_info:
        doc_count = adapter.count()
        st.metric("Fragmentos indexados", doc_count)
        st.caption(
            f"Modelo: {vec_config.embedding_model} | "
            f"Colección: {vec_config.collection_name} | "
            f"Chunks: {vec_config.chunk_size} chars, overlap {vec_config.chunk_overlap}"
        )
    with col_actions:
        if doc_count > 0:
            if st.button("Vaciar coleccion y biblioteca", type="secondary"):
                adapter.reset()
                for f in _get_docs_dir().iterdir():
                    if f.is_file():
                        f.unlink()
                st.session_state.pop("rag_search_results", None)
                st.rerun()

    st.divider()

    # --- Biblioteca de archivos ---
    library = _list_library()
    st.subheader(f"Biblioteca ({len(library)} archivo{'s' if len(library) != 1 else ''})")
    if library:
        import pandas as pd

        df = pd.DataFrame([
            {"Archivo": f["nombre"], "Tipo": f["extension"], "Tamano (KB)": f["tamano_kb"], "Fecha": f["fecha"]}
            for f in library
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

        col_del_select, col_del_btn = st.columns([3, 1])
        with col_del_select:
            to_delete = st.multiselect(
                "Seleccionar archivos para eliminar",
                options=[f["nombre"] for f in library],
                key="lib_delete_select",
            )
        with col_del_btn:
            if to_delete and st.button("Eliminar seleccionados"):
                for name in to_delete:
                    target = _get_docs_dir() / name
                    if target.exists():
                        target.unlink()
                st.info(
                    f"{len(to_delete)} archivo(s) eliminado(s) de la biblioteca. "
                    "Nota: los fragmentos ya vectorizados permanecen en la coleccion. "
                    "Usa 'Vaciar coleccion' si deseas reiniciar."
                )
                st.rerun()
    else:
        st.info("La biblioteca esta vacia. Sube archivos para comenzar.")

    st.divider()

    # --- Subida de archivos ---
    st.subheader("Subir documentos")
    st.caption(f"Formatos soportados: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

    uploaded_files = st.file_uploader(
        "Arrastra o selecciona archivos",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
        key="rag_uploader",
    )

    if uploaded_files and st.button("Vectorizar archivos", type="primary"):
        progress_bar = st.progress(0, text="Preparando...")
        results = []
        total = len(uploaded_files)
        for i, uploaded in enumerate(uploaded_files):
            progress_bar.progress(
                (i) / total,
                text=f"Procesando {uploaded.name} ({i + 1}/{total})...",
            )
            file_bytes = uploaded.read()
            _save_uploaded_file(file_bytes, uploaded.name)
            result = ingest_bytes(file_bytes, uploaded.name, adapter, vec_config)
            results.append(result)
            if debug:
                st.sidebar.caption(
                    f"{uploaded.name}: {'OK' if result.success else 'ERROR'} "
                    f"({result.chunks_indexed} chunks)"
                )

        progress_bar.progress(1.0, text="Completado")

        ok_count = sum(1 for r in results if r.success)
        total_chunks = sum(r.chunks_indexed for r in results)
        errors = [r for r in results if not r.success]

        st.success(
            f"{ok_count}/{total} archivos procesados correctamente. "
            f"Total de fragmentos indexados: {total_chunks}."
        )
        if errors:
            for err in errors:
                st.error(f"{err.filename}: {err.error}")

        with st.expander("Detalle por archivo", expanded=False):
            for r in results:
                status = "OK" if r.success else "ERROR"
                st.text(f"[{status}] {r.filename} - {r.chunks_indexed} fragmentos")
                if r.error:
                    st.caption(f"  Error: {r.error}")

    st.divider()

    # --- Búsqueda en documentos ---
    st.subheader("Buscar en documentos")
    if adapter.count() == 0:
        st.info("No hay documentos indexados. Sube archivos para comenzar a buscar.")
        return

    query = st.text_input(
        "Consulta",
        placeholder="Escribe tu pregunta sobre los documentos...",
        key="rag_query",
    )
    top_k = st.slider("Resultados a mostrar", min_value=1, max_value=20, value=vec_config.top_k)

    if st.button("Buscar"):
        if not query or not query.strip():
            st.warning("Escribe una consulta.")
            return
        with st.spinner("Buscando fragmentos relevantes..."):
            results = adapter.query(query.strip(), top_k=top_k)
            st.session_state.rag_search_results = results

    if "rag_search_results" in st.session_state and st.session_state.rag_search_results:
        results = st.session_state.rag_search_results
        st.caption(f"{len(results)} resultado(s) encontrado(s)")
        for i, r in enumerate(results, 1):
            score = 1 - r["distance"] if r["distance"] <= 1 else r["distance"]
            meta = r["metadata"]
            source = meta.get("source", "desconocido")
            chunk_idx = meta.get("chunk_index", "?")
            pages = meta.get("pages")
            label = f"#{i} | {source} (chunk {chunk_idx})"
            if pages:
                label += f" | pag. {pages}"
            label += f" | similitud: {score:.3f}"
            with st.expander(label, expanded=(i == 1)):
                st.markdown(r["text"])
                if debug:
                    st.json(meta)


def main() -> None:
    st.set_page_config(page_title="Framework DSPy", page_icon=None, layout="wide")

    if "mensaje_draft" not in st.session_state:
        st.session_state.mensaje_draft = ""
    if st.session_state.get("clear_input_next"):
        st.session_state.mensaje_draft = ""
        st.session_state.clear_input_next = False

    config = load_config()
    agent_description = load_agent_description()
    conv_config = load_conversation_config()
    api_key_ok = bool(config.api_key and config.api_key.strip())

    debug = st.sidebar.checkbox(
        "Modo debug", value=False,
        help="Muestra estado del backend y traceback completo si hay error.",
    )
    if debug:
        st.sidebar.caption("Estado: config cargada")

    tab_agent, tab_docs = st.tabs(["Agente", "RAG - Documentos"])

    with tab_agent:
        tab_agente(config, conv_config, agent_description, api_key_ok, debug)

    with tab_docs:
        tab_rag(debug)


if __name__ == "__main__":
    main()
