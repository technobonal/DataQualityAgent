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

FIX aplicado: session.rate_limiter ahora es un campo real de
SessionMemory (ver core/memory.py) en vez de un atributo dinámico
agregado acá -- ya no hace falta crearlo a mano en
create_agent_session(). Además, el chequeo de presupuesto y el
registro de tokens usados ya NO se hacen acá: chat_completion() (en
core/llm_client.py) los maneja internamente por modelo, así que
hacerlo también acá duplicaría el conteo de tokens registrados. Esto
también corrige un bug latente: con USE_LOCAL=true, 'tokens_estimados'
nunca se definía pero se usaba igual más abajo -> NameError.

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
import os
from typing import Any

import pandas as pd

from core.llm_client import (
    chat_completion, extract_text, extract_tool_calls, message_to_dict,
    LLMClientError,
)
from core.memory import SessionMemory, create_session, add_message, get_llm_messages, record_action
from tools.tool_definitions import TOOLS_SCHEMA, dispatch_tool
from security.validators import escanear_dataframe_por_inyeccion
from rag.ingest import ingest_document
from rag.validator import validar_documento, filtrar_por_relevancia
from rag.vectorstore import store_chunks, query_dictionary

# Cuántos elementos como máximo de una lista se le mandan al LLM dentro del
# resultado de una tool (ej. duplicate_indices puede tener miles de filas
# en un dataset real — el LLM solo necesita el conteo, no la lista entera).
_MAX_ITEMS_EN_LISTA_PARA_LLM = 10


def _recortar_para_llm(obj: Any) -> Any:
    """
    Recorre un resultado de tool (dict/list anidado) y recorta cualquier
    lista con más de _MAX_ITEMS_EN_LISTA_PARA_LLM elementos, dejando una
    nota de cuántos se omitieron. Evita gastar tokens de más en listas
    largas (ej. índices de filas duplicadas) que el LLM no necesita ver
    completas para explicarle el resultado al usuario — la lista completa
    sigue disponible para Streamlit a partir del dict original sin recortar.
    """
    if isinstance(obj, dict):
        return {k: _recortar_para_llm(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > _MAX_ITEMS_EN_LISTA_PARA_LLM:
            resto = len(obj) - _MAX_ITEMS_EN_LISTA_PARA_LLM
            return obj[:_MAX_ITEMS_EN_LISTA_PARA_LLM] + [f"... y {resto} más (omitido para ahorrar tokens)"]
        return [_recortar_para_llm(v) for v in obj]
    return obj

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
- CRÍTICO: cuando menciones un valor numérico específico (un promedio, un \
conteo, un porcentaje, un límite, etc.), citalo EXACTAMENTE como aparece en \
el resultado JSON que te devolvió la tool (role='tool') — nunca lo \
aproximes ni lo calcules de memoria. Por ejemplo, en el resultado de \
'tratar_valores_faltantes', el campo 'fill_values' tiene el valor REAL \
usado para rellenar cada columna, y 'values_filled' tiene la CANTIDAD de \
valores rellenados — son datos distintos, no los confundas entre sí.
- Los parámetros de una tool que NO aparecen en su lista "required" son \
OPCIONALES y ya tienen un comportamiento por defecto sensato (leé su \
"description" para saber cuál). NO le pidas al usuario que especifique un \
parámetro opcional a menos que el pedido sea genuinamente ambiguo sobre \
CUÁL sería el valor correcto. Por ejemplo: 'tratar_duplicados' NO necesita \
que le digas qué columnas mirar — por defecto compara la fila completa, \
que es lo esperado en la gran mayoría de los casos. Si el usuario pide \
"eliminá los duplicados" sin más detalle, ejecutá la tool directamente con \
sus valores por defecto — no le devuelvas la pregunta.
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


def ingest_data_dictionary(file_path: str, collection_name: str = "diccionario_datos") -> dict[str, Any]:
    """
    Ingresa un diccionario de datos (PDF, Word, Excel o CSV) al RAG:
    lo extrae (rag/ingest.py), lo valida (rag/validator.py) y, si pasa
    la validación, lo guarda en el vector store (rag/vectorstore.py).

    Es un paso OPCIONAL y separado de subir el dataset en sí — el
    usuario puede no tener un diccionario de datos, y el agente sigue
    funcionando igual sin él (ver _obtener_contexto_rag).

    Devuelve un dict con el resultado de la validación y, si se guardó,
    el resumen de cuántos chunks se indexaron.
    """
    chunks = ingest_document(file_path)
    validacion = validar_documento(chunks)

    resultado: dict[str, Any] = {"validacion": validacion.to_dict(), "guardado": None}

    if validacion.is_valid:
        resumen = store_chunks(chunks, collection_name=collection_name)
        resultado["guardado"] = resumen.to_dict()

    return resultado


def _obtener_contexto_rag(texto_usuario: str, collection_name: str = "diccionario_datos") -> str | None:
    """
    Busca en el diccionario de datos (si hay uno cargado) contexto
    relevante para lo que preguntó el usuario, y lo filtra por
    relevancia real (rag/validator.py) antes de devolverlo.

    Devuelve None — nunca lanza una excepción — si: no hay
    COHERE_API_KEY configurada, todavía no se ingirió ningún
    diccionario, o nada de lo recuperado resultó relevante. En
    cualquiera de esos casos el chat sigue funcionando igual, sin ese
    contexto extra: el RAG es un complemento opcional, nunca una
    dependencia dura del agente.
    """
    if not os.getenv("COHERE_API_KEY"):
        return None

    try:
        resultados = query_dictionary(texto_usuario, collection_name=collection_name)
    except Exception:
        return None

    if not resultados:
        return None

    filtrado = filtrar_por_relevancia(resultados)
    if not filtrado.relevant_chunks:
        return None

    lineas = [f"- {c['text']}" for c in filtrado.relevant_chunks]
    return "Contexto del diccionario de datos (puede ser útil para responder):\n" + "\n".join(lineas)


def create_agent_session(df: pd.DataFrame) -> SessionMemory:
    """
    Crea una sesión nueva para un dataset recién subido, con el system
    prompt del agente. session.rate_limiter ya viene armado de fábrica
    (ver el default_factory en SessionMemory, core/memory.py) -- no
    hace falta crearlo acá.
    """
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

    # Escaneo de seguridad, una sola vez al subir el archivo: valores de
    # celda que parezcan intentos de prompt injection (ver
    # security/validators.py — relevante porque profiler.py incluye
    # muestras de valores reales en los resultados que ve el LLM).
    alertas_inyeccion = escanear_dataframe_por_inyeccion(session.current_df)
    resultado["seguridad"] = {
        "columnas_con_alerta": {col: r.to_dict() for col, r in alertas_inyeccion.items()},
    }

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

    if alertas_inyeccion:
        resumen += (
            f"\n\n⚠️ Aviso de seguridad: se encontró contenido sospechoso (posible intento de "
            f"manipular al asistente) en {len(alertas_inyeccion)} columna(s): "
            f"{', '.join(alertas_inyeccion.keys())}. Ese contenido va a ser tratado como dato, no como instrucción."
        )

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
            result_summary=resultado,  # completo, sin recortar: esto es lo que ve Streamlit/el log de auditoría
            new_df=df_nuevo,
        )
        # Al LLM sí le mandamos la versión recortada (listas largas truncadas)
        # para no gastar tokens de más en algo que no necesita ver completo.
        contenido = _recortar_para_llm(resultado)
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

    El chequeo de presupuesto de tokens/min y el registro de uso real ya
    NO se hacen acá -- chat_completion() (core/llm_client.py) los
    maneja internamente, modelo por modelo, usando session.rate_limiter.
    Si toda la cadena de Groq está sin cupo, chat_completion() cae solo
    a Ollama local en vez de frenar acá con un mensaje de espera.

    Devuelve el texto final para mostrarle al usuario. El estado
    actualizado del dataset queda en session.current_df.
    """
    add_message(session, role="user", content=texto_usuario)

    contexto_rag = _obtener_contexto_rag(texto_usuario)
    if contexto_rag:
        add_message(session, role="system", content=contexto_rag)

    for _ in range(_MAX_TOOL_ITERATIONS):
        llm_messages = get_llm_messages(session)

        try:
            response = chat_completion(
                llm_messages,
                rate_limiter=session.rate_limiter,
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