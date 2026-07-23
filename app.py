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
from core.llm_client import USE_LOCAL
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
    if USE_LOCAL:
        st.caption("🖥️ Modo: Ollama local (más lento, sin costo)")
    else:
        st.caption("☁️ Modo: Groq (rápido)")
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
def _extraer_tabla_de_accion(sesion, accion) -> tuple[str, "pd.DataFrame"] | None:
    """
    Si una acción del agente identificó filas específicas que vale la
    pena mostrar como tabla (ej. duplicados), la extrae. Devuelve
    (titulo, dataframe) o None si esta acción no tiene una tabla asociada.
    """
    if accion.tool_name == "identificar_duplicados":
        indices = accion.result_summary.get("duplicate_indices", [])
        indices_validos = [i for i in indices if i in sesion.current_df.index]
        if indices_validos:
            titulo = f"🔁 Filas duplicadas encontradas ({len(indices_validos)}):"
            return titulo, sesion.current_df.loc[indices_validos]
    return None


def _detectar_tabla_por_palabra_clave(sesion, texto_usuario: str) -> tuple[str, "pd.DataFrame"] | None:
    """
    No depende de que el LLM decida llamar a la tool correcta — para
    algo tan concreto como "mostrame las filas duplicadas", conviene
    calcularlo directo acá si el mensaje del usuario lo menciona,
    independientemente de qué haya hecho el modelo en ese turno (a
    veces el LLM contesta solo con texto, sin volver a llamar la tool,
    si ya había mencionado el dato en un mensaje anterior del diagnóstico).
    """
    texto_lower = texto_usuario.lower()
    if any(palabra in texto_lower for palabra in ("duplicad", "repetid")):
        resultado, _ = dispatch_tool("identificar_duplicados", {}, sesion.current_df)
        indices = resultado.get("duplicate_indices", [])
        indices_validos = [i for i in indices if i in sesion.current_df.index]
        if indices_validos:
            titulo = f"🔁 Filas duplicadas encontradas ({len(indices_validos)}):"
            return titulo, sesion.current_df.loc[indices_validos]
    return None


def _parsear_estrategia_duplicados(texto: str) -> str:
    """Infiere la estrategia de tratar_duplicados a partir de las palabras del usuario."""
    texto_lower = texto.lower()
    if "última" in texto_lower or "ultima" in texto_lower:
        return "keep_last"
    if "todas" in texto_lower or "ninguna" in texto_lower or "remove_all" in texto_lower:
        return "remove_all"
    return "keep_first"  # default razonable: conservar la primera aparición


def _ejecutar_fallback_duplicados(sesion, texto_usuario: str, tools_ya_llamadas: set) -> str | None:
    """
    Respaldo determinístico: si el usuario pidió eliminar duplicados y el
    LLM NO llamó a 'tratar_duplicados' en este turno (por ejemplo, si pidió
    aclaraciones innecesarias sobre un parámetro que en realidad es
    opcional), lo ejecutamos directamente con una estrategia razonable
    inferida del texto del pedido.
    """
    texto_lower = texto_usuario.lower()
    pide_eliminar = any(p in texto_lower for p in ("elimin", "sacá", "saca", "borrá", "borra", "quitá", "quita"))
    menciona_duplicados = any(p in texto_lower for p in ("duplicad", "repetid"))

    if pide_eliminar and menciona_duplicados and "tratar_duplicados" not in tools_ya_llamadas:
        strategy = _parsear_estrategia_duplicados(texto_usuario)
        resultado, df_nuevo = dispatch_tool("tratar_duplicados", {"strategy": strategy}, sesion.current_df)
        record_action(
            sesion, tool_name="tratar_duplicados",
            arguments={"strategy": strategy}, result_summary=resultado, new_df=df_nuevo,
        )
        return f"🔧 Ejecutado automáticamente: se eliminaron {resultado['rows_removed']} fila(s) duplicada(s) (estrategia: {strategy})."
    return None


with tab_chat:
    if "chat_tables" not in st.session_state:
        st.session_state.chat_tables = {}
    if "chat_fallback_msgs" not in st.session_state:
        st.session_state.chat_fallback_msgs = {}

    for idx, m in enumerate(sesion.messages):
        if m.get("role") in ("user", "assistant") and m.get("content"):
            with st.chat_message(m["role"]):
                st.write(m["content"])
                if idx in st.session_state.chat_tables:
                    titulo, tabla = st.session_state.chat_tables[idx]
                    st.caption(titulo)
                    st.dataframe(tabla, width="stretch")
                if idx in st.session_state.chat_fallback_msgs:
                    st.info(st.session_state.chat_fallback_msgs[idx])

    prompt = st.chat_input("Escribí qué querés hacer con tus datos...")
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            acciones_antes = len(sesion.actions_log)
            mensaje_spinner = (
                "🧠 Analizando tu pedido (modo local — puede tardar unos segundos más)..."
                if USE_LOCAL else
                "🧠 Analizando tu pedido..."
            )
            with st.spinner(mensaje_spinner):
                try:
                    respuesta = send_message(sesion, prompt)
                except Exception as e:
                    respuesta = f"Ocurrió un error inesperado: {e}"
            st.write(respuesta)

            # Buscamos una tabla para mostrar por dos caminos:
            # 1) lo que el LLM efectivamente ejecutó en este turno, y
            # 2) un chequeo directo por palabra clave, que no depende de
            #    que el modelo haya decidido llamar la tool correcta
            #    (a veces no lo hace si cree que ya "sabe" la respuesta
            #    de un mensaje anterior del diagnóstico).
            indice_mensaje = len(sesion.messages) - 1
            tabla_extraida = None

            for accion in sesion.actions_log[acciones_antes:]:
                tabla_extraida = _extraer_tabla_de_accion(sesion, accion)
                if tabla_extraida:
                    break

            if not tabla_extraida:
                tabla_extraida = _detectar_tabla_por_palabra_clave(sesion, prompt)

            if tabla_extraida:
                titulo, tabla = tabla_extraida
                st.caption(titulo)
                st.dataframe(tabla, width="stretch")
                st.session_state.chat_tables[indice_mensaje] = (titulo, tabla)

            # Respaldo: si pediste eliminar duplicados y el LLM no lo hizo
            # (ej. pidió aclaraciones de más), lo ejecutamos directo acá.
            tools_llamadas_este_turno = {a.tool_name for a in sesion.actions_log[acciones_antes:]}
            mensaje_fallback = _ejecutar_fallback_duplicados(sesion, prompt, tools_llamadas_este_turno)
            if mensaje_fallback:
                st.info(mensaje_fallback)
                st.session_state.chat_fallback_msgs[indice_mensaje] = mensaje_fallback
        st.rerun()
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