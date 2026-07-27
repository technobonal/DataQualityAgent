"""
ui/chat_tab.py

Tab de Chat — lenguaje natural, el agente decide qué tool(s) usar.

Incluye lógica de:
  - Extracción de tablas de resultados de tools (ej. duplicados)
  - Detección por palabra clave (fallback si el LLM no llama la tool)
  - Fallback determinístico para eliminar duplicados
"""

from __future__ import annotations

import streamlit as st

from core.agent import send_message
from core.llm_client import USE_LOCAL
from core.memory import record_action
from tools.tool_definitions import dispatch_tool
from ui.state import StateManager
from quality.profiler import make_arrow_safe



# ─── Helpers ─────────────────────────────────────────────────

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
    calcularlo directo acá si el mensaje del usuario lo menciona.
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
    return "keep_first"  # default razonable


def _ejecutar_fallback_duplicados(sesion, texto_usuario: str, tools_ya_llamadas: set) -> str | None:
    """
    Respaldo determinístico: si el usuario pidió eliminar duplicados y el
    LLM NO llamó a 'tratar_duplicados' en este turno, lo ejecutamos
    directamente con una estrategia razonable inferida del texto.
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


# ─── Main render ─────────────────────────────────────────────

def render(manager: StateManager) -> None:
    """Renderiza la pestaña de Chat."""
    sesion = manager.get_session()

    if sesion is None:
        st.info("👆 Subí un archivo para empezar a chatear.")
        return

    # Mostrar historial de mensajes
    chat_tables = manager.get_chat_tables()
    chat_fallbacks = manager.get_chat_fallback_msgs()

    for idx, m in enumerate(sesion.messages):
        if m.get("role") in ("user", "assistant") and m.get("content"):
            with st.chat_message(m["role"]):
                st.write(m["content"])
                if idx in chat_tables:
                    titulo, tabla = chat_tables[idx]
                    st.caption(titulo)
                    st.dataframe(make_arrow_safe(tabla), width="stretch")
                if idx in chat_fallbacks:
                    st.info(chat_fallbacks[idx])

    # Input de nuevo mensaje
    prompt = st.chat_input("Escribí qué querés hacer con tus datos…")
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            acciones_antes = len(sesion.actions_log)
            mensaje_spinner = (
                "🧠 Analizando tu pedido (modo local — puede tardar unos segundos más)…"
                if USE_LOCAL else
                "🧠 Analizando tu pedido…"
            )
            with st.spinner(mensaje_spinner):
                try:
                    respuesta = send_message(sesion, prompt)
                except Exception as e:
                    respuesta = f"Ocurrió un error inesperado: {e}"
            st.write(respuesta)

            # Buscar tabla para mostrar: primero por acciones del LLM,
            # luego por palabra clave como fallback
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
                st.dataframe(make_arrow_safe(tabla), width="stretch")
                manager.set_chat_table(indice_mensaje, titulo, tabla)

            # Respaldo: si pidió eliminar duplicados y el LLM no lo hizo
            tools_llamadas_este_turno = {a.tool_name for a in sesion.actions_log[acciones_antes:]}
            mensaje_fallback = _ejecutar_fallback_duplicados(sesion, prompt, tools_llamadas_este_turno)
            if mensaje_fallback:
                st.info(mensaje_fallback)
                manager.set_chat_fallback_msg(indice_mensaje, mensaje_fallback)

        st.rerun()