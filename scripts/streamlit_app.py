"""App Streamlit para probar el wrapper DSPy: enviar texto y ver la respuesta.

Para ver más logs de Streamlit en la terminal, ejecuta:
  streamlit run scripts/streamlit_app.py --logger.level=debug
"""

import sys
import traceback
from pathlib import Path

# Asegurar que el proyecto sea importable al ejecutar: streamlit run scripts/streamlit_app.py
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

import streamlit as st

from core.config import load_config, load_conversation_config
from core.wrapper import DSPyWrapper


@st.cache_resource
def get_wrapper() -> DSPyWrapper:
    """Crea el wrapper una sola vez y lo cachea (incluye memoria si conversation.enabled)."""
    return DSPyWrapper()


def main() -> None:
    st.set_page_config(page_title="Framework DSPy", page_icon=None, layout="centered")
    st.title("Envía un mensaje al modelo")

    if "mensaje_draft" not in st.session_state:
        st.session_state.mensaje_draft = ""
    if st.session_state.get("clear_input_next"):
        st.session_state.mensaje_draft = ""
        st.session_state.clear_input_next = False

    config = load_config()
    conv_config = load_conversation_config()
    api_key_ok = bool(config.api_key and config.api_key.strip())
    st.caption(
        f"Config: {config.provider} | API key {'configurada' if api_key_ok else 'no configurada'} | Modelo: {config.model}"
    )
    if conv_config.enabled:
        st.caption("Modo conversación: el modelo usa el historial reciente.")

    debug = st.sidebar.checkbox("Modo debug", value=False, help="Muestra estado del backend y traceback completo si hay error.")
    if debug:
        st.sidebar.caption("Estado: config cargada, obteniendo wrapper...")

    try:
        wrapper = get_wrapper()
    except Exception as e:
        st.error(f"Error al crear el wrapper: {e}")
        if debug:
            st.sidebar.code(traceback.format_exc())
        return

    if debug:
        st.sidebar.caption(f"Wrapper listo. Conversación: {wrapper.has_conversation()}")
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
                if debug:
                    st.sidebar.caption("Error capturado (ver traceback arriba).")

    if st.session_state.last_response is not None:
        st.divider()
        st.subheader("Respuesta")
        st.write(st.session_state.last_response)


if __name__ == "__main__":
    main()
