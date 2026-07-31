"""
ui/analisis_tab.py

Tab de Análisis libre — a diferencia del Chat (core/agent.py, limitado
a las 15 tools fijas de limpieza/calidad), acá el usuario puede pedir
CUALQUIER cálculo/análisis sobre el dataset en lenguaje natural
("¿cuál es el promedio de ventas por región?", "¿qué producto vendió
más en marzo?"), y analysis/pandasai_wrapper.py genera el código
pandas correspondiente y lo corre en security/sandbox.py.

Nota sobre el historial de esta pestaña: a diferencia de ui/chat_tab.py
(que guarda todo en sesion.messages, porque ahí sí importa el contexto
conversacional turno a turno para el LLM), acá cada pregunta es
independiente -- no hace falta arrastrar historial de una pregunta a la
siguiente para que el análisis funcione. Por eso el historial visual de
esta pestaña vive en st.session_state, separado de sesion.messages, y
no consume tokens de contexto del chat principal.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from analysis.pandasai_wrapper import analizar_con_pandasai
from core.llm_client import USE_LOCAL
from ui.state import StateManager
from ui.components import render_dataframe_con_buscador

_KEY_HISTORIAL = "analisis_libre_historial"


def _historial_de_esta_sesion() -> list[dict]:
    """
    Devuelve (creando si hace falta) la lista de preguntas/respuestas ya
    hechas en esta pestaña. Vive en st.session_state -- se resetea sola
    cuando Streamlit reinicia la sesión del navegador, igual que el
    resto del estado de la UI.
    """
    if _KEY_HISTORIAL not in st.session_state:
        st.session_state[_KEY_HISTORIAL] = []
    return st.session_state[_KEY_HISTORIAL]


def _renderizar_resultado(resultado_output, key: str) -> None:
    """
    El `output` de un SandboxResult puede ser casi cualquier cosa (un
    número, un string, una Series, un DataFrame chico) según qué haya
    calculado el código generado -- se elige cómo mostrarlo según el tipo.
    """
    if isinstance(resultado_output, pd.DataFrame):
        render_dataframe_con_buscador(resultado_output, key=key)
    elif isinstance(resultado_output, pd.Series):
        render_dataframe_con_buscador(resultado_output.to_frame(), key=key)
    else:
        st.write(resultado_output)


def render(manager: StateManager) -> None:
    """Renderiza la pestaña de Análisis libre."""
    sesion = manager.get_session()

    if sesion is None:
        st.info("👆 Subí un archivo para empezar a analizar.")
        return

    st.caption(
        "Pedí cualquier cálculo o análisis sobre tus datos en lenguaje natural. "
        "A diferencia del Chat, acá no hay una lista fija de acciones — el "
        "código se genera al momento para responder exactamente tu pregunta."
    )

    historial = _historial_de_esta_sesion()

    for item in historial:
        with st.chat_message("user"):
            st.write(item["pregunta"])
        with st.chat_message("assistant"):
            if item["resultado"].success:
                _renderizar_resultado(item["resultado"].output, key=item["key"])
            elif item["resultado"].rejected_by_static_analysis:
                st.error(
                    "⚠️ El código generado fue rechazado por seguridad "
                    "(intentó hacer algo no permitido). No se ejecutó nada."
                )
            else:
                st.warning(f"No pude resolver esa consulta: {item['resultado'].error}")

    pregunta = st.chat_input("Preguntá algo sobre tus datos… (ej. 'promedio de ventas por región')")

    if pregunta:
        with st.chat_message("user"):
            st.write(pregunta)
        with st.chat_message("assistant"):
            mensaje_spinner = (
                "🧮 Generando y ejecutando el análisis (modo local — puede tardar más)…"
                if USE_LOCAL else
                "🧮 Generando y ejecutando el análisis…"
            )
            with st.spinner(mensaje_spinner):
                try:
                    resultado = analizar_con_pandasai(sesion.current_df, pregunta, sesion.rate_limiter)
                except Exception as e:
                    st.error(f"Ocurrió un error inesperado: {e}")
                    return

            key = f"analisis_tabla_{len(historial)}"

            if resultado.success:
                _renderizar_resultado(resultado.output, key=key)
            elif resultado.rejected_by_static_analysis:
                st.error(
                    "⚠️ El código generado fue rechazado por seguridad "
                    "(intentó hacer algo no permitido). No se ejecutó nada."
                )
            else:
                st.warning(f"No pude resolver esa consulta: {resultado.error}")

            historial.append({"pregunta": pregunta, "resultado": resultado, "key": key})

        st.rerun()