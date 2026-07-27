"""
ui/state.py

State Manager — capa delgada de acceso al estado de la sesión.

No reemplaza core.memory.SessionMemory.
Solo encapsula st.session_state y expone una interfaz limpia
para que las tabs y componentes no toquen session_state directamente.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from core.agent import create_agent_session, run_diagnosis
from core.llm_client import USE_LOCAL
from core.memory import SessionMemory


_KEY_SESSION = "session"
_KEY_DIAGNOSTICO = "diagnostico"
_KEY_IS_PROCESSING = "is_processing"
_KEY_CHAT_TABLES = "chat_tables"
_KEY_CHAT_FALLBACK_MSGS = "chat_fallback_msgs"


class StateManager:
    """Wrapper controlado sobre st.session_state."""

    def ensure_initialized(self) -> None:
        """Garantiza que las claves mínimas existan en session_state."""
        if _KEY_SESSION not in st.session_state:
            st.session_state[_KEY_SESSION] = None
        if _KEY_DIAGNOSTICO not in st.session_state:
            st.session_state[_KEY_DIAGNOSTICO] = None
        if _KEY_IS_PROCESSING not in st.session_state:
            st.session_state[_KEY_IS_PROCESSING] = False
        if _KEY_CHAT_TABLES not in st.session_state:
            st.session_state[_KEY_CHAT_TABLES] = {}
        if _KEY_CHAT_FALLBACK_MSGS not in st.session_state:
            st.session_state[_KEY_CHAT_FALLBACK_MSGS] = {}

    # --- Sesión del agente ---
    def get_session(self) -> SessionMemory | None:
        return st.session_state.get(_KEY_SESSION)

    def set_session(self, session: SessionMemory | None) -> None:
        st.session_state[_KEY_SESSION] = session

    def has_session(self) -> bool:
        return self.get_session() is not None

    # --- Diagnóstico ---
    def get_diagnostico(self) -> dict[str, Any] | None:
        return st.session_state.get(_KEY_DIAGNOSTICO)

    def set_diagnostico(self, diagnostico: dict[str, Any] | None) -> None:
        st.session_state[_KEY_DIAGNOSTICO] = diagnostico

    # --- Procesamiento (anti-superposición) ---
    def is_processing(self) -> bool:
        return bool(st.session_state.get(_KEY_IS_PROCESSING, False))

    def set_processing(self, value: bool) -> None:
        st.session_state[_KEY_IS_PROCESSING] = value

    # --- Tablas y fallbacks del chat ---
    def get_chat_tables(self) -> dict[int, Any]:
        return st.session_state.get(_KEY_CHAT_TABLES, {})

    def set_chat_table(self, message_index: int, titulo: str, tabla: Any) -> None:
        tables = self.get_chat_tables()
        tables[message_index] = (titulo, tabla)
        st.session_state[_KEY_CHAT_TABLES] = tables

    def get_chat_fallback_msgs(self) -> dict[int, str]:
        return st.session_state.get(_KEY_CHAT_FALLBACK_MSGS, {})

    def set_chat_fallback_msg(self, message_index: int, mensaje: str) -> None:
        msgs = self.get_chat_fallback_msgs()
        msgs[message_index] = mensaje
        st.session_state[_KEY_CHAT_FALLBACK_MSGS] = msgs

    # --- Subida de archivo ---
    def handle_file_upload(self, archivo) -> None:
        """Lee CSV/Excel, crea la sesión, corre el diagnóstico inicial."""
        if archivo is None or self.has_session():
            return
        df = (
            pd.read_csv(archivo)
            if archivo.name.endswith(".csv")
            else pd.read_excel(archivo)
        )
        with st.spinner("Analizando tu archivo…"):
            self.set_session(create_agent_session(df))
            self.set_diagnostico(run_diagnosis(self.get_session()))
        st.rerun()

    # --- Sidebar ---
    def render_sidebar(self) -> None:
        """Sidebar con info de la sesión y botón de reset."""
        sesion = self.get_session()
        if sesion is None:
            return
        with st.sidebar:
            st.subheader("Sesión actual")
            st.write(
                f"**Filas:** {len(sesion.current_df)}  |  "
                f"**Columnas:** {len(sesion.current_df.columns)}"
            )
            if USE_LOCAL:
                st.caption("🖥️ Modo: Ollama local (más lento, sin costo)")
            else:
                st.caption("☁️ Modo: Groq (rápido)")
            if st.button("🗑️ Empezar de nuevo con otro archivo"):
                self.reset()

    # --- Reset ---
    def reset(self) -> None:
        """Borra toda la sesión para empezar de cero."""
        st.session_state[_KEY_SESSION] = None
        st.session_state[_KEY_DIAGNOSTICO] = None
        st.session_state[_KEY_IS_PROCESSING] = False
        st.session_state[_KEY_CHAT_TABLES] = {}
        st.session_state[_KEY_CHAT_FALLBACK_MSGS] = {}
        st.rerun()