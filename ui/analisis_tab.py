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

NUEVO -- soporte de CORRECCIÓN además de análisis: hasta ahora esta tab
solo hacía preguntas de solo lectura (analizar_con_pandasai). Ahora
también puede recibir pedidos de acción ("corregí las columnas que
tienen fechas guardadas como texto"), enrutados a
corregir_columnas_con_pandasai (analysis/pandasai_wrapper.py).

Diseño Human-in-the-Loop, igual que cleaning/: una corrección generada
NUNCA se aplica sola al DataFrame de la sesión. Se muestra un preview
del cambio de dtypes (antes/después) y el usuario tiene que confirmar
con un botón antes de que sesion.current_df se actualice. Mientras esa
confirmación está pendiente, se guarda en st.session_state (no en el
historial normal) porque Streamlit re-ejecuta todo el script en cada
interacción y necesita sobrevivir al rerun.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from analysis.pandasai_wrapper import (
    analizar_con_pandasai,
    corregir_columnas_con_pandasai,
    _resumir_resultado_previo,
)
from core.agent import run_diagnosis
from core.llm_client import USE_LOCAL
from ui.state import StateManager
from ui.components import render_dataframe_con_buscador

_KEY_HISTORIAL = "analisis_libre_historial"

# NUEVO -- guarda una corrección ya generada pero todavía NO confirmada
# por el usuario. Vive aparte del historial normal porque no es algo
# que ya haya pasado -- es una propuesta de cambio pendiente de acción.
_KEY_CORRECCION_PENDIENTE = "analisis_libre_correccion_pendiente"

# NUEVO -- palabras clave simples para distinguir un PEDIDO DE ACCIÓN
# ("corregí la columna X") de una PREGUNTA de análisis ("¿cuántos nulos
# tiene la columna X?"). Heurística intencionalmente simple -- si en tu
# uso real da falsos positivos/negativos, se puede reemplazar por una
# clasificación con el LLM (prompt corto que devuelva "pregunta" o
# "correccion"), pero eso consume una llamada extra a la cadena de
# Groq/Ollama por cada mensaje, así que se arranca con esto.
_PALABRAS_CLAVE_CORRECCION = (
    "corregí", "corregir", "corrige", "convertí", "convertir", "convierte",
    "arreglá", "arreglar", "arregla", "transformá", "transformar",
    "cambiá el tipo", "cambiar el tipo", "pasá a numérico", "pasar a numérico",
    "pasá a fecha", "pasar a fecha",
)


def _parece_pedido_de_correccion(pregunta: str) -> bool:
    """Heurística simple por palabras clave -- ver nota arriba."""
    texto = pregunta.strip().lower()
    return any(palabra in texto for palabra in _PALABRAS_CLAVE_CORRECCION)


def _contexto_previo_desde_historial(historial: list[dict]) -> str | None:
    """
    NUEVO -- arma el contexto para preguntas de seguimiento ("esas
    columnas", "esos valores") a partir del último ítem EXITOSO del
    historial. Si el último ítem fue un error/rechazo, no tiene sentido
    pasarlo como contexto (no hay nada útil que resumir), así que se
    ignora y se sigue buscando hacia atrás uno que sí haya funcionado.
    """
    for item in reversed(historial):
        resultado = item["resultado"]
        if resultado.success:
            return _resumir_resultado_previo(item["pregunta"], resultado.output)
    return None


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


def _renderizar_preview_correccion(df_actual: pd.DataFrame, df_nuevo: pd.DataFrame) -> None:
    """
    NUEVO -- Muestra qué columnas cambiaron de tipo antes de aplicar la
    corrección. Si ninguna columna cambió de dtype (el LLM "corrigió"
    pero no cambió nada), se avisa igual -- puede ser una señal de que
    el pedido no hizo lo que el usuario esperaba.
    """
    comparacion = pd.DataFrame({
        "antes": df_actual.dtypes.astype(str),
        "después": df_nuevo.dtypes.astype(str),
    })
    cambios = comparacion[comparacion["antes"] != comparacion["después"]]

    if cambios.empty:
        st.info("El código corrió sin errores, pero ninguna columna cambió de tipo.")
    else:
        st.write("**Cambios de tipo propuestos:**")
        st.dataframe(cambios)


def _manejar_correccion_pendiente(manager: StateManager, sesion) -> None:
    """
    NUEVO -- Si hay una corrección generada esperando confirmación, la
    muestra con botones Aplicar/Descartar. Se llama en CADA render de la
    tab (no solo cuando llega un mensaje nuevo), porque el estado
    pendiente tiene que sobrevivir a los reruns de Streamlit.
    """
    pendiente = st.session_state.get(_KEY_CORRECCION_PENDIENTE)
    if pendiente is None:
        return

    with st.chat_message("assistant"):
        st.write(f"Propuesta de corrección para: _{pendiente['instruccion']}_")
        _renderizar_preview_correccion(sesion.current_df, pendiente["df_nuevo"])

        col_aplicar, col_descartar = st.columns(2)
        if col_aplicar.button("✅ Aplicar corrección", key="aplicar_correccion_analisis"):
            sesion.current_df = pendiente["df_nuevo"]
            # IMPORTANTE: el diagnóstico (tab Diagnóstico) se calculó una
            # sola vez al subir el archivo (ver ui/state.py,
            # handle_file_upload) y vive aparte en session_state -- si no
            # se recalcula acá, esa tab seguiría mostrando el problema de
            # tipos ya corregido como si siguiera sin resolver.
            manager.set_diagnostico(run_diagnosis(sesion))
            del st.session_state[_KEY_CORRECCION_PENDIENTE]
            st.rerun()
        if col_descartar.button("❌ Descartar", key="descartar_correccion_analisis"):
            del st.session_state[_KEY_CORRECCION_PENDIENTE]
            st.rerun()


def render(manager: StateManager) -> None:
    """Renderiza la pestaña de Análisis libre."""
    sesion = manager.get_session()

    if sesion is None:
        st.info("👆 Subí un archivo para empezar a analizar.")
        return

    # NOTA: se usa st.markdown con color forzado en vez de st.caption()
    # -- st.caption() renderiza en gris tenue por defecto, y si el CSS
    # custom (assets/style.css) pisa el color de fondo o de texto con
    # algo similar, el texto queda invisible aunque esté ahí en el HTML.
    # Forzar el color acá lo hace independiente de esa colisión.
    st.markdown(
        "<p style='color: var(--text-color, #6b7280); font-size: 0.9rem;'>"
        "Pedí cualquier cálculo o análisis sobre tus datos en lenguaje natural, "
        "o pedí una corrección de tipos (ej. 'corregí las columnas con fechas "
        "guardadas como texto'). A diferencia del Chat, acá no hay una lista "
        "fija de acciones — el código se genera al momento."
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
            elif item["resultado"].rejected_by_static_analysis:
                st.error(
                    "⚠️ El código generado fue rechazado por seguridad "
                    "(intentó hacer algo no permitido). No se ejecutó nada."
                )
            else:
                st.warning(f"No pude resolver esa consulta: {item['resultado'].error}")

    # NUEVO -- si hay una corrección esperando que el usuario la confirme
    # o descarte, se muestra ACÁ, después del historial y antes del input
    # nuevo, para que no se pierda entre reruns.
    _manejar_correccion_pendiente(manager, sesion)

    pregunta = st.chat_input(
        "Preguntá algo o pedí una corrección… (ej. 'promedio de ventas por región' "
        "o 'corregí las columnas con fechas como texto')"
    )

    if pregunta:
        with st.chat_message("user"):
            st.write(pregunta)

        es_correccion = _parece_pedido_de_correccion(pregunta)

        with st.chat_message("assistant"):
            if es_correccion:
                mensaje_spinner = (
                    "🔧 Generando la corrección (modo local — puede tardar más)…"
                    if USE_LOCAL else
                    "🔧 Generando la corrección…"
                )
                with st.spinner(mensaje_spinner):
                    try:
                        resultado = corregir_columnas_con_pandasai(
                            sesion.current_df, pregunta, sesion.rate_limiter,
                            contexto_previo=_contexto_previo_desde_historial(historial),
                        )
                    except Exception as e:
                        st.error(f"Ocurrió un error inesperado: {e}")
                        return

                if resultado.success:
                    # NO se aplica todavía -- queda pendiente de confirmación,
                    # se muestra recién en el próximo rerun (_manejar_correccion_pendiente).
                    st.session_state[_KEY_CORRECCION_PENDIENTE] = {
                        "instruccion": pregunta,
                        "df_nuevo": resultado.output,
                    }
                else:
                    # IMPORTANTE: si NO tuvo éxito (rechazo o error), no se
                    # puede mostrar con st.error/st.warning acá y listo --
                    # el st.rerun() de más abajo borra cualquier cosa que no
                    # esté guardada en algo que sobreviva al rerun. Por eso
                    # se agrega al mismo `historial` que ya usan las
                    # preguntas de análisis -- el loop de arriba ya sabe
                    # distinguir success/rejected_by_static_analysis/error
                    # y lo va a volver a pintar correctamente.
                    historial.append({
                        "pregunta": pregunta,
                        "resultado": resultado,
                        "key": f"correccion_fallida_{len(historial)}",
                    })

            else:
                mensaje_spinner = (
                    "🧮 Generando y ejecutando el análisis (modo local — puede tardar más)…"
                    if USE_LOCAL else
                    "🧮 Generando y ejecutando el análisis…"
                )
                with st.spinner(mensaje_spinner):
                    try:
                        resultado = analizar_con_pandasai(
                            sesion.current_df, pregunta, sesion.rate_limiter,
                            contexto_previo=_contexto_previo_desde_historial(historial),
                        )
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