"""
ui/components.py

Componentes reutilizables de UI para DataQualityAgent.
Widgets y helpers compartidos por múltiples tabs.

NUEVO: render_dataframe_con_buscador() -- selector de columna por nombre
(un st.selectbox arriba de la tabla) que usa el parámetro nativo
column_order de st.dataframe para traer la columna elegida al frente,
sin tener que scrollear entre las ~20 columnas del dataset a simple
vista. No arma una tabla custom -- column_order ya resuelve esto.
render_data_preview() ahora usa este selector por defecto.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from core.memory import SessionMemory, undo_last_action, reset_to_original, record_action
from tools.tool_definitions import dispatch_tool
from quality.profiler import make_arrow_safe

_OPCION_SIN_FILTRO = "(todas, orden original)"


# ─── Widgets ─────────────────────────────────────────────────

def multiselect_opcional(label: str, opciones: list[str], key: str) -> list[str] | None:
    """Multiselect donde 'sin selección' significa None (aplicar a todas), no lista vacía."""
    seleccion = st.multiselect(label, opciones, key=key)
    return seleccion if seleccion else None


def render_dataframe_con_buscador(
    df: pd.DataFrame,
    key: str,
    height: int | str | None = None,
    resaltar_nulos: bool = False,
) -> None:
    """
    Muestra un DataFrame con un buscador de columna por NOMBRE arriba de
    la tabla (un selectbox), útil en datasets con muchas columnas donde
    encontrar una a simple vista implica scrollear de más.

    Al elegir una columna en el selector, se usa el parámetro nativo
    column_order de st.dataframe para traerla al frente (primera
    posición visible, sin scroll) -- no se reordena el DataFrame en sí,
    solo cómo se ve en esta tabla puntual.

    key: identificador único del widget -- OBLIGATORIO y debe ser
         distinto en cada lugar donde se use esta función en la misma
         corrida de Streamlit (ej. "diagnostico_preview",
         f"chat_tabla_{indice_mensaje}"), o Streamlit tira error de
         key duplicada.
    height: se pasa tal cual a st.dataframe. En versiones recientes de
            Streamlit, height=None ya no es válido -- si no se
            especifica, no se pasa el parámetro (usa el default de
            Streamlit) en vez de forzar None.
    resaltar_nulos: si True, pinta el fondo de las celdas vacías (NaN)
                    en vez de mostrar la tabla plana -- útil cuando la
                    tabla mezcla columnas con y sin nulos y no alcanza
                    con verla entera para saber cuál celda puntual está
                    vacía en cada fila.

    """
    columnas = df.columns.tolist()

    columna_elegida = st.selectbox(
        "🔍 Buscar columna",
        options=[_OPCION_SIN_FILTRO] + columnas,
        key=f"{key}_buscador_columna",
    )

    if columna_elegida != _OPCION_SIN_FILTRO:
        column_order = [columna_elegida] + [c for c in columnas if c != columna_elegida]
    else:
        column_order = None  # orden original del DataFrame
    df_seguro = make_arrow_safe(df)

    # st.dataframe acepta un Styler igual que un DataFrame plano -- el
    # column_order sigue funcionando porque opera sobre las columnas
    # subyacentes, no sobre el estilo.
    tabla = (
        df_seguro.style.map(lambda v: "background-color: #5b2333" if pd.isna(v) else "")
        if resaltar_nulos
        else df_seguro
    )

    kwargs: dict[str, Any] = {"width": "stretch", "column_order": column_order}
    if height is not None:
        kwargs["height"] = height

    st.dataframe(tabla, **kwargs)


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
    """Vista rápida del DataFrame actual, con buscador de columna por nombre."""
    render_dataframe_con_buscador(sesion.current_df, key="diagnostico_preview")