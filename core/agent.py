"""
core/agent.py

El orquestador: conecta core/memory.py (estado de la sesión),
core/llm_client.py (la conexión con Groq) y tools/tool_definitions.py
(las 15 tools de limpieza/calidad) en un solo flujo conversacional.

Dos puntos de entrada principales:
  - run_diagnosis(session): corre el diagnóstico inicial DIRECTO, sin
    pasar por el LLM (es determinístico, no hace falta "razonar" para
    esto — más rápido y no gasta cuota de Groq).
  - send_message(session, texto): el chat libre — el LLM decide qué
    tool(s) llamar según lo que pida el usuario, en lenguaje natural.

Nota de seguridad: dispatch_tool() solo puede ejecutar las 15 funciones
ya escritas y probadas en cleaning/ y quality/ — el LLM elige CUÁL de
esas 15 llamar y con qué argumentos, pero nunca puede hacer que se
ejecute código arbitrario. Eso es lo que lo distingue de dejarle a un
LLM generar y correr Python libremente (como sí hace PandasAI en
analysis/pandasai_wrapper.py, que por eso necesita su propio sandbox —
ver security/sandbox.py).
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from core.llm_client import chat_completion, extract_text, extract_tool_calls, message_to_dict, LLMClientError
from core.memory import SessionMemory, create_session, add_message, get_llm_messages, record_action
from tools.tool_definitions import TOOLS_SCHEMA, dispatch_tool

# Cuántas idas y vueltas de "el LLM pide una tool -> se ejecuta -> el LLM
# ve el resultado" se permiten como máximo antes de cortar. Evita loops
# infinitos si el modelo se queda pidiendo tools sin nunca dar una
# respuesta final.
_MAX_TOOL_ITERATIONS = 5

_SYSTEM_PROMPT = """\
Sos un agente experto en calidad de datos (data wrangling), que ayuda a \
personas no técnicas a limpiar y preparar archivos Excel/CSV.

Tenés acceso a un conjunto fijo de herramientas (tools) para: identificar \
y tratar valores faltantes, corregir formato de datos, limpiar nombres/texto, \
eliminar duplicados, estandarizar, normalizar, discretizar (binning), crear \
variables indicadoras, identificar y tratar outliers, y generar un \
diagnóstico de calidad con recomendaciones.

Reglas importantes:
- Usá SIEMPRE una tool para ejecutar una acción sobre los datos. Nunca \
inventes que "ya lo hiciste" sin haber llamado a la tool correspondiente.
- Si el pedido del usuario es ambiguo (ej. no especifica qué columna), \
preguntá antes de ejecutar una tool que podría afectar columnas equivocadas.
- Explicá los resultados de cada tool en lenguaje simple, sin jerga \
técnica innecesaria — el usuario no tiene por qué saber qué es un IQR o \
un z-score en detalle.
- Si el usuario pregunta algo que ninguna tool puede resolver, decilo \
con claridad en vez de inventar una respuesta.
- No tenés capacidad de ejecutar código Python arbitrario — solo podés \
usar las tools disponibles.
"""


def create_agent_session(df: pd.DataFrame) -> SessionMemory:
    """Crea una sesión nueva para un dataset recién subido, con el system prompt del agente."""
    return create_session(df, system_prompt=_SYSTEM_PROMPT)


def run_diagnosis(session: SessionMemory) -> dict[str, Any]:
    """
    Corre el diagnóstico inicial (diagnosticar_y_recomendar) directo,
    sin pasar por el LLM — es determinístico, así que no hace falta
    "razonar" para esto, y evita gastar una llamada a Groq apenas se
    sube el archivo.

    Además dejá un resumen en el historial de la conversación (como
    mensaje del assistant), para que si el usuario después le pregunta
    algo al LLM libremente, ya tenga contexto de lo que se encontró sin
    tener que volver a correr el diagnóstico.

    Devuelve el dict completo {"diagnostico": ..., "recomendacion": ...}
    — pensado para que Streamlit lo renderice directamente (tabla,
    gráfico, lista de acciones necesarias vs. no necesarias, etc.).
    """
    resultado, _ = dispatch_tool("diagnosticar_y_recomendar", {}, session.current_df)

    score = resultado["diagnostico"]["quality_score"]
    acciones = resultado["recomendacion"]["needed_actions"]

    if acciones:
        lineas = [f"- {a['reason']}" for a in acciones]
        resumen = (
            f"Diagnóstico inicial: score de calidad {score}/100. "
            f"Encontré {len(acciones)} cosa(s) para revisar:\n" + "\n".join(lineas)
        )
    else:
        resumen = f"Diagnóstico inicial: score de calidad {score}/100. No encontré problemas relevantes."

    add_message(session, role="assistant", content=resumen)
    return resultado


def _ejecutar_tool_call(
    session: SessionMemory,
    tool_call: dict[str, Any],
) -> str:
    """
    Ejecuta una tool call concreta contra el dataset actual de la sesión,
    registra la acción en memoria (para el log y para poder deshacer), y
    devuelve el contenido (como string JSON) que se le manda de vuelta
    al LLM en un mensaje role='tool'.

    Si la tool falla (argumentos inválidos, columna inexistente, etc.),
    no se rompe la conversación: se le informa el error al LLM en el
    mismo formato, para que pueda pedir disculpas o corregir el pedido.
    """
    nombre = tool_call["name"]
    argumentos = tool_call["arguments"]

    try:
        resultado, df_nuevo = dispatch_tool(nombre, argumentos, session.current_df)
        record_action(
            session,
            tool_name=nombre,
            arguments=argumentos,
            result_summary=resultado,
            new_df=df_nuevo,
        )
        contenido = resultado
    except Exception as e:
        contenido = {"error": f"La tool '{nombre}' falló: {e}"}

    return json.dumps(contenido, ensure_ascii=False, default=str)


def send_message(session: SessionMemory, texto_usuario: str) -> str:
    """
    Punto de entrada del chat libre: agrega el mensaje del usuario al
    historial, le pregunta a Groq qué hacer (con las 15 tools
    disponibles), ejecuta las tools que el LLM pida, le devuelve los
    resultados, y repite hasta que el LLM da una respuesta final en
    texto (o hasta _MAX_TOOL_ITERATIONS, lo que pase primero).

    Devuelve el texto final para mostrarle al usuario. El estado
    actualizado del dataset queda en session.current_df.
    """
    add_message(session, role="user", content=texto_usuario)

    for _ in range(_MAX_TOOL_ITERATIONS):
        try:
            response = chat_completion(
                get_llm_messages(session),
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
            )
        except LLMClientError as e:
            return f"No pude conectarme con el modelo: {e}"

        # Se guarda el mensaje del assistant tal cual lo devolvió el LLM
        # (con su tool_calls en formato OpenAI original), para que el
        # historial que después se le manda a Groq sea válido.
        session.messages.append(message_to_dict(response))

        tool_calls = extract_tool_calls(response)
        if not tool_calls:
            return extract_text(response) or ""

        for tc in tool_calls:
            contenido_tool = _ejecutar_tool_call(session, tc)
            add_message(
                session,
                role="tool",
                content=contenido_tool,
                tool_call_id=tc["id"],
                name=tc["name"],
            )

    return (
        "No pude terminar de resolver tu pedido después de varios intentos. "
        "¿Podés reformularlo de forma más específica?"
    )


if __name__ == "__main__":
    # Prueba manual — requiere GROQ_API_KEY real en el .env.
    # IMPORTANTE: no se puede correr desde el sandbox de Claude (sin
    # acceso a api.groq.com) — la lógica del loop ya se probó con
    # respuestas simuladas (ver conversación). Correr esto desde tu
    # máquina para confirmar el flujo completo con Groq real.
    df_ejemplo = pd.DataFrame({
        "id": [1, 2, 2, 3, None],
        "edad": [22, 25, 25, None, 47],
    })

    sesion = create_agent_session(df_ejemplo)

    print("=== Diagnóstico inicial ===")
    diagnostico = run_diagnosis(sesion)
    print(f"Score: {diagnostico['diagnostico']['quality_score']}")

    print("\n=== Chat libre ===")
    respuesta = send_message(sesion, "Rellename los nulos de edad con la media, por favor")
    print(respuesta)
    print("\nDataset actual:")
    print(sesion.current_df)