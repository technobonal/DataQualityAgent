"""
ui/herramientas_tab.py

Tab de Herramientas manuales — un input dedicado para cada una de las 15
tools, para control preciso sin depender del LLM.

NUEVO — Renombrar columna / Agregar columna nueva: a diferencia de las
15 tools de arriba (que se aplican directo con un solo click, vía
ejecutar_manual + tool_result_block), estas dos van por un flujo HITL
de dos pasos (preview → Aplicar/Descartar), mismo patrón que
ui/analisis_tab.py. Motivo: es una decisión explícita del usuario, no
una inconsistencia -- las 15 tools existentes ya tienen "Deshacer" como
red de seguridad; acá se pidió ver el cambio antes de confirmarlo.

Ambas funciones (renombrar_columna, agregar_columna_vacia) viven en
analysis/pandasai_wrapper.py, NO en tools/tool_definitions.py -- son
determinísticas (sin LLM) pero conceptualmente son transformaciones de
columnas, mismo lugar que corregir_columnas/eliminar_columnas. Por eso
no pasan por dispatch_tool() ni por ejecutar_manual().

FIX -- refresh de diagnóstico: hasta esta revisión, ni las 15 tools
existentes ni las 2 nuevas actualizaban manager.set_diagnostico()
después de aplicar un cambio (a diferencia de ui/analisis_tab.py, que
sí lo hace en cada _manejar_*_pendiente) -- el score de calidad mostrado
en Diagnóstico quedaba desactualizado tras usar cualquier herramienta
manual hasta que el usuario forzara un refresh por otro lado. Ahora se
refresca en los 3 puntos de confirmación: una vez en render() para las
15 tools (cubre todos los branches sin repetir la llamada en cada uno),
y una vez en cada _manejar_*_pendiente para renombrar/agregar columna.
"""

from __future__ import annotations

import streamlit as st

from analysis.pandasai_wrapper import (
    agregar_columna_vacia,
    renombrar_columna,
    resumen_columna_agregada,
    resumen_renombrado_columna,
)
from core.agent import run_diagnosis
from core.memory import record_action
from ui.state import StateManager
from ui.components import multiselect_opcional, ejecutar_manual, tool_result_block

_KEY_RENOMBRAR_PENDIENTE = "herramientas_renombrar_pendiente"
_KEY_AGREGAR_COLUMNA_PENDIENTE = "herramientas_agregar_columna_pendiente"


def _manejar_renombrar_pendiente(manager: StateManager, sesion) -> None:
    pendiente = st.session_state.get(_KEY_RENOMBRAR_PENDIENTE)
    if pendiente is None:
        return
    st.info(
        f"Se va a renombrar **{pendiente['columna_actual']}** → "
        f"**{pendiente['nuevo_nombre']}**"
    )
    col_aplicar, col_descartar = st.columns(2)
    if col_aplicar.button("✅ Aplicar renombrado", key="aplicar_renombrar_col"):
        record_action(
            sesion,
            tool_name="renombrar_columna",
            arguments={
                "columna_actual": pendiente["columna_actual"],
                "nuevo_nombre": pendiente["nuevo_nombre"],
            },
            result_summary=pendiente["resumen"],
            new_df=pendiente["df_nuevo"],
        )
        manager.set_diagnostico(run_diagnosis(sesion))
        del st.session_state[_KEY_RENOMBRAR_PENDIENTE]
        st.rerun()
    if col_descartar.button("❌ Descartar", key="descartar_renombrar_col"):
        del st.session_state[_KEY_RENOMBRAR_PENDIENTE]
        st.rerun()


def _manejar_agregar_columna_pendiente(manager: StateManager, sesion) -> None:
    pendiente = st.session_state.get(_KEY_AGREGAR_COLUMNA_PENDIENTE)
    if pendiente is None:
        return
    st.info(f"Se va a agregar la columna **{pendiente['nombre_columna']}** ({pendiente['resumen']['posicion']})")
    col_aplicar, col_descartar = st.columns(2)
    if col_aplicar.button("✅ Aplicar columna nueva", key="aplicar_agregar_col"):
        record_action(
            sesion,
            tool_name="agregar_columna_vacia",
            arguments={
                "nombre_columna": pendiente["nombre_columna"],
                "entre_columnas": pendiente["entre_columnas"],
            },
            result_summary=pendiente["resumen"],
            new_df=pendiente["df_nuevo"],
        )
        manager.set_diagnostico(run_diagnosis(sesion))
        del st.session_state[_KEY_AGREGAR_COLUMNA_PENDIENTE]
        st.rerun()
    if col_descartar.button("❌ Descartar", key="descartar_agregar_col"):
        del st.session_state[_KEY_AGREGAR_COLUMNA_PENDIENTE]
        st.rerun()


def render(manager: StateManager) -> None:
    """Renderiza la pestaña de Herramientas manuales."""
    sesion = manager.get_session()
    if sesion is None:
        st.info("👆 Subí un archivo para usar las herramientas.")
        return

    st.caption("Ejecutá cualquier tool directamente, sin pasar por el chat.")

    # Preview pendiente de renombrado/agregado de columna (si hay) va
    # antes del selector, para que no se pierda al reinteractuar con
    # el resto del formulario.
    _manejar_renombrar_pendiente(manager, sesion)
    _manejar_agregar_columna_pendiente(manager, sesion)

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
        "Renombrar columna",
        "Agregar columna nueva",
    ])

    resultado_ejecucion = None

    # ── 1. Identificar valores faltantes ─────────────────────
    if herramienta == "Identificar valores faltantes":
        if st.button("Ejecutar"):
            resultado_ejecucion = ejecutar_manual("identificar_valores_faltantes", {}, sesion)

    # ── 2. Tratar valores faltantes ──────────────────────────
    elif herramienta == "Tratar valores faltantes":
        cols = multiselect_opcional("Columnas (vacío = todas)", columnas, "mv_cols")
        strategy = st.selectbox("Estrategia", ["mean", "median", "mode", "constant", "ffill", "bfill", "drop_rows", "drop_columns"])
        constant_value = None
        if strategy == "constant":
            constant_value = st.text_input("Valor constante")
        if st.button("Ejecutar"):
            args = {"strategy": strategy, "columns": cols}
            if constant_value is not None:
                args["constant_value"] = constant_value
            resultado_ejecucion = ejecutar_manual("tratar_valores_faltantes", args, sesion)

    # ── 3. Corregir formato de datos ─────────────────────────
    elif herramienta == "Corregir formato de datos":
        cols = multiselect_opcional("Columnas (vacío = todas)", columnas, "fmt_cols")
        if st.button("Ejecutar"):
            resultado_ejecucion = ejecutar_manual("corregir_formato_datos", {"columns": cols}, sesion)

    # ── 4. Limpiar nombres de columnas ───────────────────────
    elif herramienta == "Limpiar nombres de columnas":
        if st.button("Ejecutar"):
            resultado_ejecucion = ejecutar_manual("limpiar_nombres_columnas", {}, sesion)

    # ── 5. Limpiar columna de texto ──────────────────────────
    elif herramienta == "Limpiar columna de texto":
        col = st.selectbox("Columna", columnas)
        case = st.selectbox("Formato", ["(solo recortar espacios)", "lower", "upper", "title"])
        if st.button("Ejecutar"):
            args = {"column": col}
            if case != "(solo recortar espacios)":
                args["case"] = case
            resultado_ejecucion = ejecutar_manual("limpiar_columna_texto", args, sesion)

    # ── 6. Identificar duplicados ────────────────────────────
    elif herramienta == "Identificar duplicados":
        subset = multiselect_opcional("Columnas a comparar (vacío = fila completa)", columnas, "dup_id_subset")
        if st.button("Ejecutar"):
            resultado_ejecucion = ejecutar_manual("identificar_duplicados", {"subset": subset}, sesion)

    # ── 7. Tratar duplicados ─────────────────────────────────
    elif herramienta == "Tratar duplicados":
        strategy = st.selectbox("Estrategia", ["keep_first", "keep_last", "remove_all"])
        subset = multiselect_opcional("Columnas a comparar (vacío = fila completa)", columnas, "dup_tr_subset")
        if st.button("Ejecutar"):
            resultado_ejecucion = ejecutar_manual("tratar_duplicados", {"strategy": strategy, "subset": subset}, sesion)

    # ── 8. Estandarizar columnas ─────────────────────────────
    elif herramienta == "Estandarizar columnas":
        cols = multiselect_opcional("Columnas numéricas (vacío = todas)", columnas_numericas, "std_cols")
        suffix = st.text_input("Sufijo para columna nueva (vacío = sobrescribe)", key="std_suffix")
        if st.button("Ejecutar"):
            resultado_ejecucion = ejecutar_manual("estandarizar_columnas", {"columns": cols, "suffix": suffix or None}, sesion)

    # ── 9. Normalizar columnas ───────────────────────────────
    elif herramienta == "Normalizar columnas":
        cols = multiselect_opcional("Columnas numéricas (vacío = todas)", columnas_numericas, "norm_cols")
        col_min, col_max = st.columns(2)
        range_min = col_min.number_input("Mínimo del rango", value=0.0)
        range_max = col_max.number_input("Máximo del rango", value=1.0)
        suffix = st.text_input("Sufijo para columna nueva (vacío = sobrescribe)", key="norm_suffix")
        if st.button("Ejecutar"):
            resultado_ejecucion = ejecutar_manual("normalizar_columnas", {
                "columns": cols, "range_min": range_min, "range_max": range_max, "suffix": suffix or None,
            }, sesion)

    # ── 10. Discretizar columna ──────────────────────────────
    elif herramienta == "Discretizar columna":
        col = st.selectbox("Columna numérica", columnas_numericas)
        method = st.selectbox("Método", ["equal_width", "equal_frequency", "custom"])
        if method == "custom":
            bins_texto = st.text_input("Límites separados por coma (ej: 0,30,60,100)")
            bins = [float(x.strip()) for x in bins_texto.split(",") if x.strip()] if bins_texto else []
        else:
            bins = st.number_input("Cantidad de bins", min_value=2, value=5, step=1)
        labels_texto = st.text_input("Nombres de los bins separados por coma (opcional)")
        labels = [x.strip() for x in labels_texto.split(",") if x.strip()] if labels_texto else None
        if st.button("Ejecutar"):
            resultado_ejecucion = ejecutar_manual("discretizar_columna", {
                "column": col, "method": method, "bins": bins, "labels": labels,
            }, sesion)

    # ── 11. Crear variables indicadoras ──────────────────────
    elif herramienta == "Crear variables indicadoras":
        columnas_texto = sesion.current_df.select_dtypes(include=["object", "string"]).columns.tolist()
        cols = multiselect_opcional("Columnas categóricas (vacío = todas las de texto)", columnas_texto, "ind_cols")
        drop_first = st.checkbox("Descartar la primera categoría (evita colinealidad)")
        top_n = st.number_input("Top N categorías (0 = sin límite)", min_value=0, value=0, step=1)
        drop_original = st.checkbox("Eliminar columna original", value=True)
        if st.button("Ejecutar"):
            resultado_ejecucion = ejecutar_manual("crear_variables_indicadoras", {
                "columns": cols, "drop_first": drop_first,
                "top_n": top_n if top_n > 0 else None, "drop_original": drop_original,
            }, sesion)

    # ── 12. Identificar outliers ─────────────────────────────
    elif herramienta == "Identificar outliers":
        cols = multiselect_opcional("Columnas numéricas (vacío = todas)", columnas_numericas, "out_id_cols")
        multiplier = st.number_input("Multiplicador IQR", min_value=0.1, value=1.5, step=0.1)
        if st.button("Ejecutar"):
            resultado_ejecucion = ejecutar_manual("identificar_outliers", {"columns": cols, "multiplier": multiplier}, sesion)

    # ── 13. Tratar outliers ──────────────────────────────────
    elif herramienta == "Tratar outliers":
        strategy = st.selectbox("Estrategia", ["cap", "remove"])
        cols = multiselect_opcional("Columnas numéricas (vacío = todas)", columnas_numericas, "out_tr_cols")
        multiplier = st.number_input("Multiplicador IQR", min_value=0.1, value=1.5, step=0.1, key="out_tr_mult")
        if st.button("Ejecutar"):
            resultado_ejecucion = ejecutar_manual("tratar_outliers", {"strategy": strategy, "columns": cols, "multiplier": multiplier}, sesion)

    # ── 14. Generar reporte de calidad ───────────────────────
    elif herramienta == "Generar reporte de calidad":
        dup_subset = multiselect_opcional("Columnas para duplicados (vacío = fila completa)", columnas, "rep_dup")
        out_cols = multiselect_opcional("Columnas para outliers (vacío = todas numéricas)", columnas_numericas, "rep_out")
        if st.button("Ejecutar"):
            resultado_ejecucion = ejecutar_manual("generar_reporte_calidad", {
                "duplicate_subset": dup_subset, "outlier_columns": out_cols,
            }, sesion)

    # ── 15. Diagnosticar y recomendar ────────────────────────
    elif herramienta == "Diagnosticar y recomendar":
        dup_subset = multiselect_opcional("Columnas para duplicados (vacío = fila completa)", columnas, "diag_dup")
        out_cols = multiselect_opcional("Columnas para outliers (vacío = todas numéricas)", columnas_numericas, "diag_out")
        if st.button("Ejecutar"):
            resultado_ejecucion = ejecutar_manual("diagnosticar_y_recomendar", {
                "duplicate_subset": dup_subset, "outlier_columns": out_cols,
            }, sesion)

    # ── 16. Renombrar columna (NUEVO, HITL: preview -> Aplicar/Descartar) ──
    elif herramienta == "Renombrar columna":
        columna_actual = st.selectbox("Columna a renombrar", columnas, key="renombrar_col_actual")
        nuevo_nombre = st.text_input("Nuevo nombre", key="renombrar_col_nuevo_nombre")
        if st.button("Renombrar columna"):
            try:
                df_nuevo = renombrar_columna(sesion.current_df, columna_actual, nuevo_nombre)
                resumen = resumen_renombrado_columna(columna_actual, nuevo_nombre.strip())
                st.session_state[_KEY_RENOMBRAR_PENDIENTE] = {
                    "columna_actual": columna_actual,
                    "nuevo_nombre": nuevo_nombre.strip(),
                    "df_nuevo": df_nuevo,
                    "resumen": resumen,
                }
                st.rerun()
            except ValueError as e:
                st.error(str(e))

    # ── 17. Agregar columna nueva (NUEVO, HITL: preview -> Aplicar/Descartar) ──
    elif herramienta == "Agregar columna nueva":
        nombre_columna_nueva = st.text_input("Nombre de la columna nueva", key="agregar_col_nombre")

        opciones_posicion = ["Al final"]
        if len(columnas) >= 2:
            opciones_posicion.append("Entre dos columnas")
        posicion = st.radio("¿Dónde va la columna?", opciones_posicion, key="agregar_col_posicion")

        entre_columnas = None
        if posicion == "Entre dos columnas":
            # Solo se puede elegir la columna IZQUIERDA -- la derecha se
            # deriva automáticamente como la siguiente columna en el
            # orden actual, así el par siempre es adyacente por
            # construcción (nunca dispara el ValueError de no-adyacencia
            # de agregar_columna_vacia()).
            columna_izquierda = st.selectbox(
                "Insertar después de la columna",
                columnas[:-1],
                key="agregar_col_izquierda",
            )
            idx = columnas.index(columna_izquierda)
            columna_derecha = columnas[idx + 1]
            st.caption(f"Va a quedar entre **{columna_izquierda}** y **{columna_derecha}**.")
            entre_columnas = (columna_izquierda, columna_derecha)

        if st.button("Agregar columna"):
            try:
                df_nuevo = agregar_columna_vacia(
                    sesion.current_df, nombre_columna_nueva, entre_columnas
                )
                resumen = resumen_columna_agregada(nombre_columna_nueva.strip(), entre_columnas)
                st.session_state[_KEY_AGREGAR_COLUMNA_PENDIENTE] = {
                    "nombre_columna": nombre_columna_nueva.strip(),
                    "entre_columnas": entre_columnas,
                    "df_nuevo": df_nuevo,
                    "resumen": resumen,
                }
                st.rerun()
            except ValueError as e:
                st.error(str(e))

    # ── Resultado (tools 1-15, ejecución directa) ────────────
    if resultado_ejecucion is not None:
        manager.set_diagnostico(run_diagnosis(sesion))
    tool_result_block(resultado_ejecucion)