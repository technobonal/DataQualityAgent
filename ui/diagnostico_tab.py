"""
ui/diagnostico_tab.py

Tab de Diagnóstico — muestra el resultado del análisis inicial:
score de calidad, acciones necesarias, alertas de seguridad,
vista del DataFrame y controles de undo/reset/download.
"""

from __future__ import annotations

import streamlit as st

from ui.state import StateManager
from ui.components import render_undo_controls, render_data_preview
from quality.profiler import make_arrow_safe



def render(manager: StateManager) -> None:
    """Renderiza la pestaña de Diagnóstico."""
    sesion = manager.get_session()
    diag = manager.get_diagnostico()

    if sesion is None or diag is None:
        st.info("👆 Subí un archivo para ver el diagnóstico.")
        return

    # --- Score de calidad ---
    col_score, col_acciones = st.columns([1, 2])
    with col_score:
        st.metric("Score de calidad", f"{diag['diagnostico']['quality_score']}/100")

    # --- Alertas de seguridad ---
    alertas_seguridad = diag.get("seguridad", {}).get("columnas_con_alerta", {})
    if alertas_seguridad:
        st.warning(
            f"⚠️ Se encontró contenido sospechoso (posible intento de manipular al asistente) "
            f"en: {', '.join(alertas_seguridad.keys())}. Se trata como dato, no como instrucción."
        )

    # --- Acciones necesarias ---
    st.subheader("✅ Acciones necesarias")
    emojis_prioridad = {"alta": "🔴", "media": "🟡", "baja": "🟢"}
    acciones = diag["recomendacion"]["needed_actions"]
    if acciones:
        for accion in acciones:
            emoji = emojis_prioridad.get(accion["priority"], "⚪")
            nombre_tool = accion["tool_name"] or "revisión manual (sin tool automática todavía)"
            st.write(f"{emoji} **{nombre_tool}** — {accion['reason']}")
    else:
        st.success("No se encontraron problemas relevantes.")

    # --- Lo que NO hace falta hacer ---
    with st.expander("Ver lo que NO hace falta hacer"):
        for s in diag["recomendacion"]["not_needed"]:
            st.write(f"✓ **{s['tool_name']}** — {s['reason']}")

    # --- Vista de datos + controles ---
    st.subheader("Vista actual de los datos")
    render_data_preview(sesion)
    render_undo_controls(sesion)