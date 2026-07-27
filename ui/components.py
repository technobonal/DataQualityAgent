

"""
ui/components.py

Componentes reutilizables de UI para DataQualityAgent.
Widgets y helpers compartidos por múltiples tabs.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from core.memory import SessionMemory, undo_last_action, reset_to_original, record_action
from tools.tool_definitions import dispatch_tool
from quality.profiler import make_arrow_safe


# ─── Widgets ─────────────────────────────────────────────────

def multiselect_opcional(label: str, opciones: list[str], key: str) -> list[str] | None:
    """Multiselect donde 'sin selección' significa None (aplicar a todas), no lista vacía."""
    seleccion = st.multiselect(label, opciones, key=key)
    return seleccion if seleccion else None


# ─── Tool execution ─────────────────────────────────────────

def ejecutar_manual(nombre_tool: str, argumentos: dict, sesion: SessionMemory) -> dict:
    """Ejecuta una tool directamente (sin pasar por el LLM) y registra la acción en memoria."""
    resultado, df_nuevo = dispatch_tool(nombre_tool, argumentos, sesion.current_df)
    record_action(
        sesion,
        tool_name=nombre_tool,
        arguments=argumentos,
        result_summary=resultado,
        new_df=df_nuevo,
    )
    return resultado


def tool_result_block(resultado: dict | None) -> None:
    """Muestra el resultado de una tool y fuerza rerun para actualizar la UI."""
    if resultado is not None:
        st.success("Ejecutado correctamente.")
        st.json(resultado)
        st.rerun()


# ─── Data controls ───────────────────────────────────────────

def render_undo_controls(sesion: SessionMemory) -> None:
    """Botones de deshacer / volver al original / descargar CSV."""
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("↩️ Deshacer último paso", use_container_width=True):
            if undo_last_action(sesion):
                st.success("Deshecho.")
                st.rerun()
            else:
                st.info("No hay nada para deshacer.")
    with c2:
        if st.button("🔄 Volver al archivo original", use_container_width=True):
            reset_to_original(sesion)
            st.rerun()
    with c3:
        csv_bytes = sesion.current_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar CSV limpio",
            csv_bytes,
            "datos_limpios.csv",
            "text/csv",
            use_container_width=True,
        )


def render_data_preview(sesion: SessionMemory) -> None:
    """Vista rápida del DataFrame actual."""
    st.dataframe(make_arrow_safe(sesion.current_df), width="stretch")