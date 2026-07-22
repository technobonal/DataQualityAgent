"""
app.py

Interfaz de Streamlit de DataQualityAgent.

Cuatro secciones:
  1. Diagnóstico — al subir el archivo, corre automáticamente y muestra
     qué hace falta limpiar (y qué NO hace falta), con un score de calidad.
  2. Chat — lenguaje natural, el agente decide qué tool(s) usar.
  3. Herramientas manuales — un input dedicado para cada una de las 15
     tools, para control preciso sin depender del LLM.
  4. Diccionario de datos — subir un documento (PDF/Word/Excel/CSV) que
     describa las columnas, para que el chat lo use como contexto (RAG).

Corré con: streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from core.agent import create_agent_session, run_diagnosis, send_message, ingest_data_dictionary
from core.memory import undo_last_action, reset_to_original, record_action
from tools.tool_definitions import dispatch_tool

st.set_page_config(page_title="DataQualityAgent", page_icon="🧹", layout="wide")

# ============================================================
# Estado de sesión (persiste entre reruns de Streamlit)
# ============================================================
if "session" not in st.session_state:
    st.session_state.session = None
if "diagnostico" not in st.session_state:
    st.session_state.diagnostico = None


def _ejecutar_manual(nombre_tool: str, argumentos: dict) -> dict:
    """Ejecuta una tool directamente (sin pasar por el LLM) y registra la acción en memoria."""
    sesion = st.session_state.session
    resultado, df_nuevo = dispatch_tool(nombre_tool, argumentos, sesion.current_df)
    record_action(sesion, tool_name=nombre_tool, arguments=argumentos, result_summary=resultado, new_df=df_nuevo)
    return resultado


def _multiselect_opcional(label: str, opciones: list[str], key: str) -> list[str] | None:
    """Multiselect donde 'sin selección' significa None (aplicar a todas), no lista vacía."""
    seleccion = st.multiselect(label, opciones, key=key)
    return seleccion if seleccion else None


st.title("🧹 DataQualityAgent")
st.caption("Subí un Excel o CSV y dejá que el agente te diga qué falta limpiar — o hacelo vos mismo, a mano.")

# ============================================================
# 1. Subida del archivo principal
# ============================================================
archivo = st.file_uploader("📂 Subí tu archivo de datos", type=["csv", "xlsx", "xls"])

if archivo is not None and st.session_state.session is None:
    df = pd.read_csv(archivo) if archivo.name.endswith(".csv") else pd.read_excel(archivo)
    with st.spinner("Analizando tu archivo..."):
        st.session_state.session = create_agent_session(df)
        st.session_state.diagnostico = run_diagnosis(st.session_state.session)
    st.rerun()

if st.session_state.session is None:
    st.info("👆 Subí un archivo para empezar.")
    st.stop()

sesion = st.session_state.session

with st.sidebar:
    st.subheader("Sesión actual")
    st.write(f"**Filas:** {len(sesion.current_df)}  |  **Columnas:** {len(sesion.current_df.columns)}")
    if st.button("🗑️ Empezar de nuevo con otro archivo"):
        st.session_state.session = None
        st.session_state.diagnostico = None
        st.rerun()

tab_diagnostico, tab_chat, tab_herramientas, tab_diccionario = st.tabs(
    ["📋 Diagnóstico", "💬 Chat", "🔧 Herramientas manuales", "📖 Diccionario de datos"]
)

# ============================================================
# TAB 1: Diagnóstico
# ============================================================
with tab_diagnostico:
    diag = st.session_state.diagnostico

    col_score, col_acciones = st.columns([1, 2])
    with col_score:
        st.metric("Score de calidad", f"{diag['diagnostico']['quality_score']}/100")

    alertas_seguridad = diag.get("seguridad", {}).get("columnas_con_alerta", {})
    if alertas_seguridad:
        st.warning(
            f"⚠️ Se encontró contenido sospechoso (posible intento de manipular al asistente) "
            f"en: {', '.join(alertas_seguridad.keys())}. Se trata como dato, no como instrucción."
        )

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

    with st.expander("Ver lo que NO hace falta hacer"):
        for s in diag["recomendacion"]["not_needed"]:
            st.write(f"✓ **{s['tool_name']}** — {s['reason']}")

    st.subheader("Vista actual de los datos")
    st.dataframe(sesion.current_df, width="stretch")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("↩️ Deshacer último paso", width="stretch"):
            if undo_last_action(sesion):
                st.success("Deshecho.")
                st.rerun()
            else:
                st.info("No hay nada para deshacer.")
    with col2:
        if st.button("🔄 Volver al archivo original", width="stretch"):
            reset_to_original(sesion)
            st.rerun()
    with col3:
        csv_bytes = sesion.current_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar CSV limpio", csv_bytes, "datos_limpios.csv", "text/csv", width="stretch")

# ============================================================
# TAB 2: Chat
# ============================================================
with tab_chat:
    for m in sesion.messages:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            with st.chat_message(m["role"]):
                st.write(m["content"])

    prompt = st.chat_input("Escribí qué querés hacer con tus datos...")
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                try:
                    respuesta = send_message(sesion, prompt)
                except Exception as e:
                    respuesta = f"Ocurrió un error inesperado: {e}"
            st.write(respuesta)
        st.rerun()

# ============================================================
# TAB 3: Herramientas manuales (una por cada una de las 15 tools)
# ============================================================
with tab_herramientas:
    st.caption("Ejecutá cualquier tool directamente, sin pasar por el chat.")

    columnas = sesion.current_df.columns.tolist()
    columnas_numericas = sesion.current_df.select_dtypes(include="number").columns.tolist()

    herramienta = st.selectbox("Elegí una herramienta", [
        "Identificar valores faltantes",
        "Tratar valores faltantes",
        "Corregir formato de datos",
        "Limpiar nombres de columnas",
        "Limpiar columna de texto",
        "Identificar duplicados",
        "Tratar duplicados",
        "Estandarizar columnas",
        "Normalizar columnas",
        "Discretizar columna",
        "Crear variables indicadoras",
        "Identificar outliers",
        "Tratar outliers",
        "Generar reporte de calidad",
        "Diagnosticar y recomendar",
    ])

    resultado_ejecucion = None

    if herramienta == "Identificar valores faltantes":
        if st.button("Ejecutar"):
            resultado_ejecucion = _ejecutar_manual("identificar_valores_faltantes", {})

    elif herramienta == "Tratar valores faltantes":
        cols = _multiselect_opcional("Columnas (vacío = todas)", columnas, "mv_cols")
        strategy = st.selectbox("Estrategia", ["mean", "median", "mode", "constant", "ffill", "bfill", "drop_rows", "drop_columns"])
        constant_value = None
        if strategy == "constant":
            constant_value = st.text_input("Valor constante")
        if st.button("Ejecutar"):
            args = {"strategy": strategy, "columns": cols}
            if constant_value is not None:
                args["constant_value"] = constant_value
            resultado_ejecucion = _ejecutar_manual("tratar_valores_faltantes", args)

    elif herramienta == "Corregir formato de datos":
        cols = _multiselect_opcional("Columnas (vacío = todas)", columnas, "fmt_cols")
        if st.button("Ejecutar"):
            resultado_ejecucion = _ejecutar_manual("corregir_formato_datos", {"columns": cols})

    elif herramienta == "Limpiar nombres de columnas":
        if st.button("Ejecutar"):
            resultado_ejecucion = _ejecutar_manual("limpiar_nombres_columnas", {})

    elif herramienta == "Limpiar columna de texto":
        col = st.selectbox("Columna", columnas)
        case = st.selectbox("Formato", ["(solo recortar espacios)", "lower", "upper", "title"])
        if st.button("Ejecutar"):
            args = {"column": col}
            if case != "(solo recortar espacios)":
                args["case"] = case
            resultado_ejecucion = _ejecutar_manual("limpiar_columna_texto", args)

    elif herramienta == "Identificar duplicados":
        subset = _multiselect_opcional("Columnas a comparar (vacío = fila completa)", columnas, "dup_id_subset")
        if st.button("Ejecutar"):
            resultado_ejecucion = _ejecutar_manual("identificar_duplicados", {"subset": subset})

    elif herramienta == "Tratar duplicados":
        strategy = st.selectbox("Estrategia", ["keep_first", "keep_last", "remove_all"])
        subset = _multiselect_opcional("Columnas a comparar (vacío = fila completa)", columnas, "dup_tr_subset")
        if st.button("Ejecutar"):
            resultado_ejecucion = _ejecutar_manual("tratar_duplicados", {"strategy": strategy, "subset": subset})

    elif herramienta == "Estandarizar columnas":
        cols = _multiselect_opcional("Columnas numéricas (vacío = todas)", columnas_numericas, "std_cols")
        suffix = st.text_input("Sufijo para columna nueva (vacío = sobrescribe)", key="std_suffix")
        if st.button("Ejecutar"):
            resultado_ejecucion = _ejecutar_manual("estandarizar_columnas", {"columns": cols, "suffix": suffix or None})

    elif herramienta == "Normalizar columnas":
        cols = _multiselect_opcional("Columnas numéricas (vacío = todas)", columnas_numericas, "norm_cols")
        col_min, col_max = st.columns(2)
        range_min = col_min.number_input("Mínimo del rango", value=0.0)
        range_max = col_max.number_input("Máximo del rango", value=1.0)
        suffix = st.text_input("Sufijo para columna nueva (vacío = sobrescribe)", key="norm_suffix")
        if st.button("Ejecutar"):
            resultado_ejecucion = _ejecutar_manual("normalizar_columnas", {
                "columns": cols, "range_min": range_min, "range_max": range_max, "suffix": suffix or None,
            })

    elif herramienta == "Discretizar columna":
        col = st.selectbox("Columna numérica", columnas_numericas)
        method = st.selectbox("Método", ["equal_width", "equal_frequency", "custom"])
        if method == "custom":
            bins_texto = st.text_input("Límites separados por coma (ej: 0,30,60,100)")
            bins = [float(x.strip()) for x in bins_texto.split(",")] if bins_texto else []
        else:
            bins = st.number_input("Cantidad de bins", min_value=2, value=5, step=1)
        labels_texto = st.text_input("Nombres de los bins separados por coma (opcional)")
        labels = [x.strip() for x in labels_texto.split(",")] if labels_texto else None
        if st.button("Ejecutar"):
            resultado_ejecucion = _ejecutar_manual("discretizar_columna", {
                "column": col, "method": method, "bins": bins, "labels": labels,
            })

    elif herramienta == "Crear variables indicadoras":
        columnas_texto = sesion.current_df.select_dtypes(include=["object", "string"]).columns.tolist()
        cols = _multiselect_opcional("Columnas categóricas (vacío = todas las de texto)", columnas_texto, "ind_cols")
        drop_first = st.checkbox("Descartar la primera categoría (evita colinealidad)")
        top_n = st.number_input("Top N categorías (0 = sin límite)", min_value=0, value=0, step=1)
        drop_original = st.checkbox("Eliminar columna original", value=True)
        if st.button("Ejecutar"):
            resultado_ejecucion = _ejecutar_manual("crear_variables_indicadoras", {
                "columns": cols, "drop_first": drop_first,
                "top_n": top_n if top_n > 0 else None, "drop_original": drop_original,
            })

    elif herramienta == "Identificar outliers":
        cols = _multiselect_opcional("Columnas numéricas (vacío = todas)", columnas_numericas, "out_id_cols")
        multiplier = st.number_input("Multiplicador IQR", min_value=0.1, value=1.5, step=0.1)
        if st.button("Ejecutar"):
            resultado_ejecucion = _ejecutar_manual("identificar_outliers", {"columns": cols, "multiplier": multiplier})

    elif herramienta == "Tratar outliers":
        strategy = st.selectbox("Estrategia", ["cap", "remove"])
        cols = _multiselect_opcional("Columnas numéricas (vacío = todas)", columnas_numericas, "out_tr_cols")
        multiplier = st.number_input("Multiplicador IQR", min_value=0.1, value=1.5, step=0.1, key="out_tr_mult")
        if st.button("Ejecutar"):
            resultado_ejecucion = _ejecutar_manual("tratar_outliers", {"strategy": strategy, "columns": cols, "multiplier": multiplier})

    elif herramienta == "Generar reporte de calidad":
        dup_subset = _multiselect_opcional("Columnas para duplicados (vacío = fila completa)", columnas, "rep_dup")
        out_cols = _multiselect_opcional("Columnas para outliers (vacío = todas numéricas)", columnas_numericas, "rep_out")
        if st.button("Ejecutar"):
            resultado_ejecucion = _ejecutar_manual("generar_reporte_calidad", {
                "duplicate_subset": dup_subset, "outlier_columns": out_cols,
            })

    elif herramienta == "Diagnosticar y recomendar":
        dup_subset = _multiselect_opcional("Columnas para duplicados (vacío = fila completa)", columnas, "diag_dup")
        out_cols = _multiselect_opcional("Columnas para outliers (vacío = todas numéricas)", columnas_numericas, "diag_out")
        if st.button("Ejecutar"):
            resultado_ejecucion = _ejecutar_manual("diagnosticar_y_recomendar", {
                "duplicate_subset": dup_subset, "outlier_columns": out_cols,
            })

    if resultado_ejecucion is not None:
        st.success("Ejecutado correctamente.")
        st.json(resultado_ejecucion)
        st.rerun()

# ============================================================
# TAB 4: Diccionario de datos (RAG)
# ============================================================
with tab_diccionario:
    st.caption(
        "Subí un documento que explique qué significa cada columna de tu dataset "
        "(PDF, Word, Excel o CSV) — el chat lo va a usar como contexto adicional."
    )

    dic_file = st.file_uploader("📖 Diccionario de datos", type=["pdf", "docx", "csv", "xlsx"], key="dic_uploader")

    if dic_file is not None:
        if st.button("Procesar e indexar"):
            extension = Path(dic_file.name).suffix
            with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp:
                tmp.write(dic_file.getvalue())
                ruta_temporal = tmp.name

            with st.spinner("Procesando el diccionario de datos..."):
                try:
                    resultado = ingest_data_dictionary(ruta_temporal)
                except Exception as e:
                    st.error(f"No se pudo procesar el documento: {e}")
                    resultado = None

            if resultado is not None:
                if resultado["validacion"]["is_valid"]:
                    st.success(
                        f"Documento indexado: {resultado['guardado']['chunks_added']} fragmento(s) guardados."
                    )
                else:
                    st.error("El documento no pasó la validación: " + "; ".join(resultado["validacion"]["issues"]))