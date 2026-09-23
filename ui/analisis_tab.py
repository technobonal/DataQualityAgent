"""
ui/analisis_tab.py — Análisis libre + intent_router integrado.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

from analysis.intent_router import (
    IntentResult,
    clasificar_intent,
    necesita_aclaracion,
)
from analysis.pandasai_wrapper import (
    analizar_con_pandasai,
    corregir_columnas_con_pandasai,
    eliminar_columnas_con_pandasai,
    eliminar_filas_con_pandasai,
    generar_grafico_con_pandasai,
    limpiar_celdas_con_pandasai,
    resumen_eliminacion_columnas,
    resumen_eliminacion_filas,
    resumen_relleno_celdas,
    _diff_dtypes,
    _resumir_resultado_previo,
)
from core.agent import run_diagnosis
from core.llm_client import USE_LOCAL
from core.memory import record_action
from ui.state import StateManager
from ui.components import render_dataframe_con_buscador

_KEY_HISTORIAL = "analisis_libre_historial"
_KEY_CORRECCION_PENDIENTE = "analisis_libre_correccion_pendiente"
_KEY_ELIMINACION_PENDIENTE = "analisis_libre_eliminacion_pendiente"
_KEY_CELDAS_PENDIENTE = "analisis_libre_celdas_pendiente"
_KEY_ELIMINACION_COLUMNAS_PENDIENTE = "analisis_libre_eliminacion_columnas_pendiente"
_KEY_ACLARACION_PENDIENTE = "analisis_libre_aclaracion_pendiente"
_KEY_TIPO_FORZADO = "_analisis_tipo_forzado"

_MAX_FILAS_PREVIEW_ELIMINADAS = 20

_PALABRAS_CLAVE_ELIMINACION = (
    "eliminá", "eliminar", "elimina", "sacá", "sacar", "saca",
    "borrá", "borrar", "borra", "quitá", "quitar", "quita",
    "remové", "remover", "remove",
)
_PALABRAS_CLAVE_FILA = ("fila", "filas")

_MAP_INTENT_A_TIPO = {
    "analizar": "analisis",
    "corregir_tipos": "correccion",
    "eliminar_filas": "eliminacion",
    "eliminar_columnas": "eliminacion_columnas",
    "limpiar_celdas": "celdas",
    "grafico": "grafico",
}


def _pedido_ambiguo_se_refiere_a_columnas(
    historial: list[dict], columnas_dataset: list[str]
) -> bool:
    columnas_set = set(columnas_dataset)
    for item in reversed(historial):
        resultado = item["resultado"]
        if not resultado.success:
            continue
        output = resultado.output
        if isinstance(output, (pd.Series, pd.Index)):
            candidatos = list(output.index) if isinstance(output, pd.Series) else list(output)
        elif isinstance(output, list):
            candidatos = output
        else:
            continue
        if (
            candidatos
            and all(isinstance(c, str) for c in candidatos)
            and set(candidatos).issubset(columnas_set)
        ):
            return True
        return False
    return False


def _tipo_desde_intent_result(
    r: IntentResult,
    pregunta: str,
    historial: list[dict],
    columnas_dataset: list[str],
) -> str | None:
    if necesita_aclaracion(r):
        texto = pregunta.strip().lower()
        menciona_elim = any(p in texto for p in _PALABRAS_CLAVE_ELIMINACION)
        if menciona_elim or (r.intent or "").startswith("eliminar"):
            if _pedido_ambiguo_se_refiere_a_columnas(historial, columnas_dataset):
                return "eliminacion_columnas"
        return None

    tipo = _MAP_INTENT_A_TIPO.get(r.intent)
    if tipo is None:
        return None

    if tipo == "eliminacion":
        texto = pregunta.strip().lower()
        menciona_fila = any(p in texto for p in _PALABRAS_CLAVE_FILA)
        if not menciona_fila and _pedido_ambiguo_se_refiere_a_columnas(
            historial, columnas_dataset
        ):
            return "eliminacion_columnas"
    return tipo


def _contexto_previo_desde_historial(historial: list[dict]) -> str | None:
    for item in reversed(historial):
        if item["resultado"].success:
            return _resumir_resultado_previo(item["pregunta"], item["resultado"].output)
    return None


def _historial_de_esta_sesion() -> list[dict]:
    if _KEY_HISTORIAL not in st.session_state:
        st.session_state[_KEY_HISTORIAL] = []
    return st.session_state[_KEY_HISTORIAL]


def _renderizar_resultado(resultado_output: Any, key: str) -> None:
    if isinstance(resultado_output, pd.DataFrame):
        render_dataframe_con_buscador(resultado_output, key=key)
    elif isinstance(resultado_output, pd.Series):
        render_dataframe_con_buscador(resultado_output.to_frame(), key=key)
    elif isinstance(resultado_output, Figure):
        st.pyplot(resultado_output)
    else:
        st.write(resultado_output)


def _renderizar_preview_correccion(df_actual: pd.DataFrame, df_nuevo: pd.DataFrame) -> None:
    cambios = _diff_dtypes(df_actual, df_nuevo)
    if not cambios:
        st.info("El código corrió sin errores, pero ninguna columna cambió de tipo.")
    else:
        st.write("**Cambios de tipo propuestos:**")
        st.dataframe(pd.DataFrame(cambios).T)


def _renderizar_preview_eliminacion(resumen: dict[str, Any]) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Filas antes", resumen["filas_antes"])
    col2.metric("Filas después", resumen["filas_despues"])
    col3.metric("Filas a eliminar", resumen["filas_eliminadas"])
    if resumen["filas_eliminadas"] == 0:
        st.info("El código corrió sin errores, pero ninguna fila cumplió el criterio.")
    else:
        st.write("**Muestra de filas que se eliminarían:**")
        st.dataframe(resumen["df_eliminadas"].head(_MAX_FILAS_PREVIEW_ELIMINADAS))


def _renderizar_preview_eliminacion_columnas(resumen: dict[str, Any]) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Columnas antes", len(resumen["columnas_antes"]))
    col2.metric("Columnas después", len(resumen["columnas_despues"]))
    col3.metric("Columnas a eliminar", len(resumen["columnas_eliminadas"]))
    if not resumen["columnas_eliminadas"]:
        st.info("El código corrió sin errores, pero ninguna columna cumplió el criterio.")
    else:
        st.write("**Columnas que se eliminarían:**")
        st.write(", ".join(resumen["columnas_eliminadas"]))


def _renderizar_preview_celdas(resumen: dict[str, Any]) -> None:
    st.metric("Celdas a modificar", resumen["celdas_afectadas_total"])
    if resumen["celdas_afectadas_total"] == 0:
        st.info("El código corrió sin errores, pero ninguna celda cumplió el criterio.")
    else:
        st.write("**Celdas afectadas por columna:**")
        st.dataframe(
            pd.DataFrame(
                {
                    "columna": list(resumen["columnas_afectadas"].keys()),
                    "celdas_modificadas": list(resumen["columnas_afectadas"].values()),
                }
            )
        )


def _metadata_para_log(resumen: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in resumen.items() if k != "df_eliminadas"}


def _manejar_correccion_pendiente(manager: StateManager, sesion) -> None:
    pendiente = st.session_state.get(_KEY_CORRECCION_PENDIENTE)
    if pendiente is None:
        return
    with st.chat_message("assistant"):
        st.write(f"Propuesta de corrección para: _{pendiente['instruccion']}_")
        _renderizar_preview_correccion(sesion.current_df, pendiente["df_nuevo"])
        col_aplicar, col_descartar = st.columns(2)
        if col_aplicar.button("✅ Aplicar corrección", key="aplicar_correccion_analisis"):
            record_action(
                sesion,
                tool_name="pandasai_correccion",
                arguments={"instruccion": pendiente["instruccion"]},
                result_summary=_diff_dtypes(sesion.current_df, pendiente["df_nuevo"]),
                new_df=pendiente["df_nuevo"],
            )
            manager.set_diagnostico(run_diagnosis(sesion))
            del st.session_state[_KEY_CORRECCION_PENDIENTE]
            st.rerun()
        if col_descartar.button("❌ Descartar", key="descartar_correccion_analisis"):
            del st.session_state[_KEY_CORRECCION_PENDIENTE]
            st.rerun()


def _manejar_eliminacion_pendiente(manager: StateManager, sesion) -> None:
    pendiente = st.session_state.get(_KEY_ELIMINACION_PENDIENTE)
    if pendiente is None:
        return
    with st.chat_message("assistant"):
        st.write(f"Propuesta de eliminación de filas para: _{pendiente['instruccion']}_")
        _renderizar_preview_eliminacion(pendiente["resumen"])
        col_aplicar, col_descartar = st.columns(2)
        if col_aplicar.button("✅ Aplicar eliminación", key="aplicar_eliminacion_analisis"):
            record_action(
                sesion,
                tool_name="pandasai_eliminacion_filas",
                arguments={"instruccion": pendiente["instruccion"]},
                result_summary=_metadata_para_log(pendiente["resumen"]),
                new_df=pendiente["df_nuevo"],
            )
            manager.set_diagnostico(run_diagnosis(sesion))
            del st.session_state[_KEY_ELIMINACION_PENDIENTE]
            st.rerun()
        if col_descartar.button("❌ Descartar", key="descartar_eliminacion_analisis"):
            del st.session_state[_KEY_ELIMINACION_PENDIENTE]
            st.rerun()


def _manejar_eliminacion_columnas_pendiente(manager: StateManager, sesion) -> None:
    pendiente = st.session_state.get(_KEY_ELIMINACION_COLUMNAS_PENDIENTE)
    if pendiente is None:
        return
    with st.chat_message("assistant"):
        st.write(f"Propuesta de eliminación de columnas para: _{pendiente['instruccion']}_")
        _renderizar_preview_eliminacion_columnas(pendiente["resumen"])
        col_aplicar, col_descartar = st.columns(2)
        if col_aplicar.button(
            "✅ Aplicar eliminación de columnas", key="aplicar_eliminacion_columnas_analisis"
        ):
            record_action(
                sesion,
                tool_name="pandasai_eliminacion_columnas",
                arguments={"instruccion": pendiente["instruccion"]},
                result_summary=pendiente["resumen"],
                new_df=pendiente["df_nuevo"],
            )
            manager.set_diagnostico(run_diagnosis(sesion))
            del st.session_state[_KEY_ELIMINACION_COLUMNAS_PENDIENTE]
            st.rerun()
        if col_descartar.button("❌ Descartar", key="descartar_eliminacion_columnas_analisis"):
            del st.session_state[_KEY_ELIMINACION_COLUMNAS_PENDIENTE]
            st.rerun()


def _manejar_celdas_pendiente(manager: StateManager, sesion) -> None:
    pendiente = st.session_state.get(_KEY_CELDAS_PENDIENTE)
    if pendiente is None:
        return
    with st.chat_message("assistant"):
        st.write(f"Propuesta de modificación de celdas para: _{pendiente['instruccion']}_")
        _renderizar_preview_celdas(pendiente["resumen"])
        col_aplicar, col_descartar = st.columns(2)
        if col_aplicar.button("✅ Aplicar cambio en celdas", key="aplicar_celdas_analisis"):
            record_action(
                sesion,
                tool_name="pandasai_celdas",
                arguments={"instruccion": pendiente["instruccion"]},
                result_summary=pendiente["resumen"],
                new_df=pendiente["df_nuevo"],
            )
            manager.set_diagnostico(run_diagnosis(sesion))
            del st.session_state[_KEY_CELDAS_PENDIENTE]
            st.rerun()
        if col_descartar.button("❌ Descartar", key="descartar_celdas_analisis"):
            del st.session_state[_KEY_CELDAS_PENDIENTE]
            st.rerun()


def _manejar_aclaracion_pendiente(manager: StateManager, sesion) -> None:
    pendiente = st.session_state.get(_KEY_ACLARACION_PENDIENTE)
    if pendiente is None:
        return
    pregunta = pendiente["pregunta"]
    with st.chat_message("assistant"):
        st.warning(f"No estoy seguro de qué acción aplicar. Pedido: _{pregunta}_")
        if pendiente.get("reason"):
            st.caption(f"Señal del clasificador: {pendiente['reason']}")
        c1, c2, c3 = st.columns(3)
        c4, c5, c6 = st.columns(3)
        eleccion = None
        if c1.button("📊 Solo analizar", key="acl_analizar"):
            eleccion = "analisis"
        if c2.button("🔧 Corregir tipos", key="acl_tipos"):
            eleccion = "correccion"
        if c3.button("🗑️ Eliminar filas", key="acl_filas"):
            eleccion = "eliminacion"
        if c4.button("🗑️ Eliminar columnas", key="acl_cols"):
            eleccion = "eliminacion_columnas"
        if c5.button("🧹 Celdas (rellenar/vaciar)", key="acl_celdas"):
            eleccion = "celdas"
        if c6.button("📈 Gráfico", key="acl_graf"):
            eleccion = "grafico"
        if st.button("❌ Cancelar", key="acl_cancel"):
            del st.session_state[_KEY_ACLARACION_PENDIENTE]
            st.rerun()
        if eleccion is not None:
            del st.session_state[_KEY_ACLARACION_PENDIENTE]
            st.session_state[_KEY_TIPO_FORZADO] = {"pregunta": pregunta, "tipo": eleccion}
            st.rerun()


def _renderizar_diagnostico(diagnostico: dict[str, Any]) -> None:
    score = diagnostico["diagnostico"]["quality_score"]
    acciones = diagnostico["recomendacion"]["needed_actions"]
    alertas = diagnostico.get("seguridad", {}).get("columnas_con_alerta", {})
    with st.expander(f"📋 Diagnóstico inicial — score de calidad: {score}/100", expanded=bool(acciones)):
        if acciones:
            st.write(f"**{len(acciones)} cosa(s) para revisar:**")
            for a in acciones:
                st.write(f"- {a['reason']}")
        else:
            st.success("No se encontraron problemas relevantes.")
        if alertas:
            st.warning(
                f"⚠️ Contenido sospechoso en {len(alertas)} columna(s): {', '.join(alertas.keys())}."
            )


def _despachar_pedido(tipo_pedido: str, pregunta: str, sesion, historial: list[dict]) -> None:
    contexto = _contexto_previo_desde_historial(historial)

    if tipo_pedido == "correccion":
        msg = "🔧 Generando la corrección…" + (" (local)" if USE_LOCAL else "")
        with st.spinner(msg):
            try:
                resultado = corregir_columnas_con_pandasai(
                    sesion.current_df, pregunta, sesion.rate_limiter, contexto_previo=contexto
                )
            except Exception as e:
                st.error(f"Ocurrió un error inesperado: {e}")
                return
        if resultado.success:
            st.session_state[_KEY_CORRECCION_PENDIENTE] = {
                "instruccion": pregunta, "df_nuevo": resultado.output
            }
        else:
            historial.append({"pregunta": pregunta, "resultado": resultado, "key": f"correccion_fallida_{len(historial)}"})

    elif tipo_pedido == "eliminacion":
        msg = "🗑️ Generando la eliminación de filas…" + (" (local)" if USE_LOCAL else "")
        with st.spinner(msg):
            try:
                resultado = eliminar_filas_con_pandasai(
                    sesion.current_df, pregunta, sesion.rate_limiter, contexto_previo=contexto
                )
            except Exception as e:
                st.error(f"Ocurrió un error inesperado: {e}")
                return
        if resultado.success:
            resumen = resumen_eliminacion_filas(sesion.current_df, resultado.output, pregunta)
            st.session_state[_KEY_ELIMINACION_PENDIENTE] = {
                "instruccion": pregunta, "df_nuevo": resultado.output, "resumen": resumen
            }
        else:
            historial.append({"pregunta": pregunta, "resultado": resultado, "key": f"eliminacion_fallida_{len(historial)}"})

    elif tipo_pedido == "eliminacion_columnas":
        msg = "🗑️ Generando la eliminación de columnas…" + (" (local)" if USE_LOCAL else "")
        with st.spinner(msg):
            try:
                resultado = eliminar_columnas_con_pandasai(
                    sesion.current_df, pregunta, sesion.rate_limiter, contexto_previo=contexto
                )
            except Exception as e:
                st.error(f"Ocurrió un error inesperado: {e}")
                return
        if resultado.success:
            resumen = resumen_eliminacion_columnas(sesion.current_df, resultado.output, pregunta)
            st.session_state[_KEY_ELIMINACION_COLUMNAS_PENDIENTE] = {
                "instruccion": pregunta, "df_nuevo": resultado.output, "resumen": resumen
            }
        else:
            historial.append({"pregunta": pregunta, "resultado": resultado, "key": f"eliminacion_columnas_fallida_{len(historial)}"})

    elif tipo_pedido == "celdas":
        msg = "🧹 Generando el cambio en celdas…" + (" (local)" if USE_LOCAL else "")
        with st.spinner(msg):
            try:
                resultado = limpiar_celdas_con_pandasai(
                    sesion.current_df, pregunta, sesion.rate_limiter, contexto_previo=contexto
                )
            except Exception as e:
                st.error(f"Ocurrió un error inesperado: {e}")
                return
        if resultado.success:
            resumen = resumen_relleno_celdas(sesion.current_df, resultado.output, pregunta)
            st.session_state[_KEY_CELDAS_PENDIENTE] = {
                "instruccion": pregunta, "df_nuevo": resultado.output, "resumen": resumen
            }
        else:
            historial.append({"pregunta": pregunta, "resultado": resultado, "key": f"celdas_fallida_{len(historial)}"})

    elif tipo_pedido == "grafico":
        msg = "📊 Generando el gráfico…" + (" (local)" if USE_LOCAL else "")
        with st.spinner(msg):
            try:
                resultado = generar_grafico_con_pandasai(
                    sesion.current_df, pregunta, sesion.rate_limiter, contexto_previo=contexto
                )
            except Exception as e:
                st.error(f"Ocurrió un error inesperado: {e}")
                return
        key = f"analisis_grafico_{len(historial)}"
        if resultado.success:
            _renderizar_resultado(resultado.output, key=key)
        elif getattr(resultado, "rejected_by_static_analysis", False):
            st.error("⚠️ Código rechazado por seguridad. No se ejecutó nada.")
        else:
            st.warning(f"No pude generar ese gráfico: {resultado.error}")
        historial.append({"pregunta": pregunta, "resultado": resultado, "key": key})

    else:
        msg = "🧮 Generando y ejecutando el análisis…" + (" (local)" if USE_LOCAL else "")
        with st.spinner(msg):
            try:
                resultado = analizar_con_pandasai(
                    sesion.current_df, pregunta, sesion.rate_limiter, contexto_previo=contexto
                )
            except Exception as e:
                st.error(f"Ocurrió un error inesperado: {e}")
                return
        key = f"analisis_tabla_{len(historial)}"
        if resultado.success:
            _renderizar_resultado(resultado.output, key=key)
        elif getattr(resultado, "rejected_by_static_analysis", False):
            st.error("⚠️ Código rechazado por seguridad. No se ejecutó nada.")
        else:
            st.warning(f"No pude resolver esa consulta: {resultado.error}")
        historial.append({"pregunta": pregunta, "resultado": resultado, "key": key})


def render(manager: StateManager) -> None:
    sesion = manager.get_session()
    if sesion is None:
        st.info("👆 Subí un archivo para empezar a analizar.")
        return

    diagnostico = manager.get_diagnostico()
    if diagnostico is not None:
        _renderizar_diagnostico(diagnostico)

    st.markdown(
        "<p style='color: var(--text-color, #6b7280); font-size: 0.9rem;'>"
        "Pedí análisis, corrección de tipos, eliminación de filas/columnas, "
        "relleno de celdas o gráficos en lenguaje natural. "
        "También podés encadenar: 'listame columnas con muchos nulos' y después 'esas, eliminalas'."
        "</p>",
        unsafe_allow_html=True,
    )

    historial = _historial_de_esta_sesion()
    for item in historial:
        with st.chat_message("user"):
            st.write(item["pregunta"])
        with st.chat_message("assistant"):
            if item["resultado"].success:
                _renderizar_resultado(item["resultado"].output, key=item["key"])
            elif getattr(item["resultado"], "rejected_by_static_analysis", False):
                st.error("⚠️ Código rechazado por seguridad. No se ejecutó nada.")
            else:
                st.warning(f"No pude resolver esa consulta: {item['resultado'].error}")

    _manejar_correccion_pendiente(manager, sesion)
    _manejar_eliminacion_pendiente(manager, sesion)
    _manejar_eliminacion_columnas_pendiente(manager, sesion)
    _manejar_celdas_pendiente(manager, sesion)
    _manejar_aclaracion_pendiente(manager, sesion)

    forzado = st.session_state.pop(_KEY_TIPO_FORZADO, None)
    if forzado is not None:
        with st.chat_message("user"):
            st.write(forzado["pregunta"])
        with st.chat_message("assistant"):
            _despachar_pedido(forzado["tipo"], forzado["pregunta"], sesion, historial)
        st.rerun()
        return

    pregunta = st.chat_input(
        "Preguntá algo o pedí una acción… (ej. promedio por región, eliminá filas sin ciudad, graficá ventas)"
    )

    if pregunta:
        with st.chat_message("user"):
            st.write(pregunta)

        columnas_dataset = list(sesion.current_df.columns)
        contexto = _contexto_previo_desde_historial(historial)
        intent_r = clasificar_intent(pregunta, sesion.rate_limiter, contexto_previo=contexto)
        tipo_pedido = _tipo_desde_intent_result(intent_r, pregunta, historial, columnas_dataset)

        if tipo_pedido is None:
            st.session_state[_KEY_ACLARACION_PENDIENTE] = {
                "pregunta": pregunta,
                "reason": (
                    f"{intent_r.intent} ({intent_r.method}, "
                    f"conf={intent_r.confidence:.2f}) {intent_r.reason}"
                ),
            }
            st.rerun()
            return

        with st.chat_message("assistant"):
            _despachar_pedido(tipo_pedido, pregunta, sesion, historial)
        st.rerun()