"""
core/memory.py

Memoria de sesión del agente: guarda estas cosas a la vez, todas atadas a
la misma sesión de un usuario trabajando sobre un archivo:

  1. El historial de la conversación (mensajes en formato OpenAI/Groq,
     para que llm_client.py tenga contexto entre turnos).
  2. El estado actual del DataFrame (qué transformaciones ya se aplicaron).
  3. Un log de acciones (qué tool se llamó, con qué argumentos y qué
     resultado dio) — sirve como auditoría y para poder deshacer pasos.
  4. Un RateLimiter propio de la sesión (security/rate_limiter.py), para
     que el cupo de tokens/minuto de Groq no se comparta entre distintos
     usuarios/sesiones trabajando al mismo tiempo -- cada sesión lleva
     su propia cuenta, igual que ya pasa con current_df o messages.

Sin dependencia de LLM en sí misma — es estado puro, testeable en
aislado. llm_client.py y agent.py son quienes la usan.

OPTIMIZACIÓN DE VELOCIDAD (histórica): _MAX_HISTORY_MESSAGES bajó de 20
a 8. Con Ollama local en CPU, el tiempo de espera está dominado por
cuánto tarda el modelo en LEER el prompt completo -- un historial largo
se vuelve a leer entero en CADA turno.

OPTIMIZACIÓN DE TOKENS (nueva): compact_completed_turn().
El problema real no era solo la CANTIDAD de mensajes, sino su PESO: cada
turno donde el LLM llama tools deja en el historial el mensaje
'assistant' con tool_calls MÁS el JSON completo de cada resultado de
tool (aunque ya recortado por _recortar_para_llm en agent.py, un
diagnóstico o reporte de calidad completo sigue siendo pesado). Ese
JSON crudo solo hace falta DENTRO del turno en que se generó, para que
el LLM pueda redactar su respuesta final en texto -- una vez que esa
respuesta en texto ya existe, los turnos futuros no necesitan releer el
JSON: la respuesta en texto ya resume lo que importa.

compact_completed_turn() reemplaza, al cerrar un turno, la secuencia
[user, assistant+tool_calls, tool, tool, ..., assistant(texto final)]
por simplemente [user, assistant(texto final)] -- msmo contenido
semántico para el usuario, sin el peso del JSON intermedio. Se llama al
final de agent.send_message(), después de tener la respuesta final.

Además, get_llm_messages() ahora recorta respetando los límites de cada
turno (nunca corta un tool_call de su respuesta), en vez de cortar por
posición fija -- evita mandar un mensaje 'tool' huérfano, que Groq
rechaza con error 400 por no tener su 'assistant' correspondiente
inmediatamente antes en la secuencia.

FIX (esta revisión) -- desincronización entre actions_log y df_history:
antes, record_action() agregaba SIEMPRE una entrada a actions_log, sin
importar si la acción modificó el dataset (new_df is not None) o fue de
solo lectura (new_df is None) -- pero df_history solo crecía en el caso
de escritura. Como undo_last_action() hacía pop() de las dos listas a
la vez asumiendo que estaban alineadas 1 a 1, una secuencia real como
[corrección (escritura), análisis (lectura)] rompía todo: al deshacer,
sacaba el snapshot correcto de df_history (el de la corrección) pero
sacaba del log la entrada del análisis (la última), no la de la
corrección -- quedaban desincronizados.

Se resolvió en la raíz, no con un parche: record_action() ahora
directamente NO registra nada en actions_log (ni en df_history) cuando
new_df es None. Decisión de diseño: actions_log es el historial de
"qué se le hizo al dataset" (auditoría + undo), no un log genérico de
toda la actividad de la sesión -- las lecturas (preguntas de análisis,
gráficos) ya se muestran en el historial visual de ui/analisis_tab.py,
no necesitan vivir acá también. Con esto, actions_log y df_history
quedan SIEMPRE 1 a 1 por construcción -- no por una convención que haya
que recordar respetar en cada call site.

NUEVO -- redo_last_action(): antes solo existía un paso atrás. Se
agregaron redo_df_stack y redo_actions_stack (listas paralelas, igual
que df_history/actions_log, pero se mantienen en lockstep porque un
único código -- undo_last_action/redo_last_action -- es responsable de
tocar ambas juntas siempre, nunca por separado). record_action() vacía
las dos pilas de redo cada vez que se aplica una acción nueva -- mismo
comportamiento estándar que cualquier editor con undo/redo: seguir
limpiando después de deshacer invalida cualquier "rehacer" que hubiera
quedado pendiente, porque ya no representa un futuro coherente del
dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from security.rate_limiter import RateLimiter

# Cuántos TURNOS completos (no mensajes sueltos) de historial se le mandan
# al LLM como máximo. Un turno = todo lo que pasó desde un mensaje 'user'
# hasta la siguiente respuesta final del assistant. Contar por turnos (y
# no por mensajes sueltos) es lo que evita partir un intercambio de tool
# calling por la mitad.
_MAX_HISTORY_TURNS = 4

# Cuántos snapshots del DataFrame se guardan para poder deshacer pasos.
# Mismo tope se usa para la pila de redo (ver redo_df_stack) -- no tendría
# sentido poder rehacer más pasos de los que se pueden deshacer.
_MAX_UNDO_STEPS = 10


@dataclass
class ActionRecord:
    """
    Registro de una acción (tool) ya aplicada al dataset en esta sesión.

    Solo existen entradas de acciones que MODIFICARON el dataset -- ver
    la nota "FIX" del docstring del módulo. Si necesitás en algún
    momento un log de TODA la actividad de la sesión (lecturas incluidas),
    eso es un log separado, a propósito no se mezcla acá.
    """
    tool_name: str
    arguments: dict[str, Any]
    result_summary: dict[str, Any]
    timestamp: str  # ISO 8601, UTC

    def to_dict(self) -> dict[str, Any]:
        """Serializa a dict plano — útil para mostrar el log en Streamlit."""
        return asdict(self)


@dataclass(eq=False)  # eq=False: un DataFrame no se puede comparar con == de forma segura
class SessionMemory:
    """
    Memoria completa de una sesión de trabajo sobre un dataset.

    original_df:  el archivo tal cual lo subió el usuario, sin tocar
                  (permite "empezar de nuevo" en cualquier momento).
    current_df:   el estado actual, después de aplicar 0 o más tools.
    messages:     historial de la conversación en formato OpenAI/Groq
                  ({"role": ..., "content": ...}, con tool_calls/tool_call_id
                  cuando corresponda).
    actions_log:  qué se le hizo al dataset, en orden, para auditoría y undo.
                  Solo acciones que modificaron el dataset (ver ActionRecord).
                  NO se toca con compact_completed_turn() -- el log de
                  auditoría siempre queda completo, sin recortar; lo único
                  que se compacta es lo que se le vuelve a mandar al LLM.
    df_history:   snapshots de current_df ANTES de cada acción de escritura,
                  para poder deshacer. Limitado a max_undo_steps para no
                  consumir memoria de más en datasets grandes. Siempre 1 a 1
                  con actions_log (misma longitud, mismo orden).
    redo_df_stack:      snapshots descartados por undo_last_action(), para
                        poder rehacerlos con redo_last_action(). Se vacía
                        por completo cada vez que se aplica una acción nueva.
    redo_actions_stack: ActionRecord correspondiente a cada entrada de
                        redo_df_stack -- misma relación 1 a 1 que
                        df_history/actions_log, mantenida en lockstep por
                        undo_last_action()/redo_last_action() (únicos
                        puntos del código que tocan estas dos listas).
    rate_limiter: cuota de tokens/minuto de Groq propia de ESTA sesión --
                  ver security/rate_limiter.py. default_factory=RateLimiter
                  para que cada sesión nueva arranque con su propio
                  presupuesto lleno, sin pisar el de otras sesiones.
    """
    original_df: pd.DataFrame
    current_df: pd.DataFrame
    messages: list[dict[str, Any]] = field(default_factory=list)
    actions_log: list[ActionRecord] = field(default_factory=list)
    df_history: list[pd.DataFrame] = field(default_factory=list)
    redo_df_stack: list[pd.DataFrame] = field(default_factory=list)
    redo_actions_stack: list[ActionRecord] = field(default_factory=list)
    rate_limiter: RateLimiter = field(default_factory=RateLimiter)
    max_history_turns: int = _MAX_HISTORY_TURNS
    max_undo_steps: int = _MAX_UNDO_STEPS


def create_session(df: pd.DataFrame, system_prompt: str | None = None) -> SessionMemory:
    """
    Crea una sesión nueva a partir del DataFrame recién subido.

    system_prompt: si se pasa, queda como el primer mensaje del historial
                   (rol 'system') — típicamente las instrucciones del
                   agente para Groq.
    """
    session = SessionMemory(original_df=df.copy(), current_df=df.copy())
    if system_prompt:
        add_message(session, role="system", content=system_prompt)
    return session


def add_message(
    session: SessionMemory,
    role: str,
    content: str | None,
    tool_calls: list[dict[str, Any]] | None = None,
    tool_call_id: str | None = None,
    name: str | None = None,
) -> None:
    """
    Agrega un mensaje al historial de la conversación.

    role: 'system', 'user', 'assistant' o 'tool'.
    tool_calls:   cuando el assistant decide invocar una o más tools
                  (formato OpenAI/Groq).
    tool_call_id: requerido cuando role='tool', para asociar la
                  respuesta con el tool_call correspondiente.
    name:         nombre de la tool, cuando role='tool'.
    """
    mensaje: dict[str, Any] = {"role": role, "content": content}
    if tool_calls is not None:
        mensaje["tool_calls"] = tool_calls
    if tool_call_id is not None:
        mensaje["tool_call_id"] = tool_call_id
    if name is not None:
        mensaje["name"] = name

    session.messages.append(mensaje)


def _indices_inicio_de_turno(mensajes: list[dict[str, Any]]) -> list[int]:
    """
    Devuelve, en orden, el índice de cada mensaje 'user' dentro de
    `mensajes` -- cada uno marca el arranque de un turno nuevo. Se usa
    tanto para recortar por turnos completos (get_llm_messages) como
    para ubicar el turno más reciente a compactar (compact_completed_turn).
    """
    return [i for i, m in enumerate(mensajes) if m.get("role") == "user"]


def get_llm_messages(session: SessionMemory) -> list[dict[str, Any]]:
    """
    Devuelve los mensajes a enviar al LLM, recortando el historial si es
    muy largo. Si el primer mensaje es el system prompt, se conserva
    siempre (no cuenta contra el límite).

    El recorte es por TURNOS completos (desde un 'user' hasta el próximo
    'user', o el final), nunca por posición fija de mensaje -- así nunca
    se corta un 'assistant' con tool_calls de su(s) mensaje(s) 'tool'
    correspondientes, que Groq rechazaría por venir sin su tool_call
    original en la misma request.
    """
    if not session.messages:
        return []

    if session.messages[0].get("role") == "system":
        system_msg, resto = session.messages[0], session.messages[1:]
    else:
        system_msg, resto = None, session.messages

    inicios = _indices_inicio_de_turno(resto)
    if len(inicios) > session.max_history_turns:
        corte = inicios[-session.max_history_turns]
        resto = resto[corte:]

    return ([system_msg] if system_msg else []) + resto


def compact_completed_turn(session: SessionMemory) -> None:
    """
    Compacta el ÚLTIMO turno del historial una vez que ya terminó (el
    LLM ya dio su respuesta final en texto, sea con o sin tools de por
    medio). Reemplaza:

        [user, assistant+tool_calls, tool, tool, ..., assistant(texto)]

    por:

        [user, assistant(texto)]

    -- el mismo contenido conversacional que ve el usuario, sin el JSON
    crudo de los resultados de tools intermedios, que ya cumplió su
    función (que el LLM redactara la respuesta) y no aporta nada a
    turnos futuros. Si el turno no usó ninguna tool (fue directo
    user -> assistant), no hay nada que compactar y la función no
    cambia nada.

    Llamar SIEMPRE al final de agent.send_message(), justo antes de
    devolver la respuesta al usuario -- nunca a mitad de un loop de
    tool calling todavía en curso (ahí el JSON intermedio todavía hace
    falta para que el LLM vea el resultado de la tool que acaba de pedir).

    No toca session.actions_log -- el log de auditoría/undo sigue
    completo siempre; esto solo afecta lo que se le vuelve a mandar al
    LLM en session.messages.
    """
    inicios = _indices_inicio_de_turno(session.messages)
    if not inicios:
        return

    inicio_ultimo_turno = inicios[-1]
    turno = session.messages[inicio_ultimo_turno:]

    # Turno sin tool calling (2 mensajes: user + assistant texto) -> nada que compactar.
    if len(turno) <= 2:
        return

    mensaje_usuario = turno[0]
    mensaje_final = turno[-1]

    # Salvaguarda: si por algún motivo el último mensaje del turno no es
    # la respuesta final del assistant (ej. el turno se cortó por
    # _MAX_TOOL_ITERATIONS sin resolver), no tocar nada -- mejor pagar
    # tokens de más que perder contexto de un turno inconcluso.
    if mensaje_final.get("role") != "assistant" or mensaje_final.get("tool_calls"):
        return

    session.messages[inicio_ultimo_turno:] = [mensaje_usuario, mensaje_final]


def record_action(
    session: SessionMemory,
    tool_name: str,
    arguments: dict[str, Any],
    result_summary: dict[str, Any],
    new_df: pd.DataFrame | None = None,
) -> None:
    """
    Registra que se aplicó una tool y, si modificó el dataset (new_df no
    es None), actualiza current_df — guardando antes un snapshot para
    poder deshacer con undo_last_action().

    FIX (ver docstring del módulo): si new_df es None (acción de solo
    lectura), la función ahora NO HACE NADA -- ni toca df_history ni
    agrega nada a actions_log. Antes sí agregaba a actions_log igual,
    lo que rompía la correspondencia 1 a 1 con df_history y causaba el
    bug de undo_last_action() sacando la entrada equivocada del log.
    Si en el futuro hace falta loguear también las lecturas, eso debe
    ser una lista/función aparte -- no reabrir esto.

    NUEVO: cualquier acción de escritura vacía las pilas de redo -- una
    vez que el usuario sigue limpiando después de un undo, el estado que
    quedó pendiente de "rehacer" ya no es un futuro coherente del
    dataset actual.
    """
    if new_df is None:
        return

    session.df_history.append(session.current_df)
    if len(session.df_history) > session.max_undo_steps:
        session.df_history.pop(0)
    session.current_df = new_df

    session.actions_log.append(
        ActionRecord(
            tool_name=tool_name,
            arguments=arguments,
            result_summary=result_summary,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
    )

    # Cualquier acción nueva invalida el "futuro" que hubiera quedado
    # pendiente de un undo anterior -- comportamiento estándar de
    # undo/redo (ver docstring del módulo).
    session.redo_df_stack.clear()
    session.redo_actions_stack.clear()


def undo_last_action(session: SessionMemory) -> bool:
    """
    Deshace el último paso que modificó el dataset (vuelve current_df al
    snapshot anterior). Devuelve True si pudo deshacer, False si no hay
    nada para deshacer.

    Nota: solo deshace cambios al DataFrame. Las tools de "solo lectura"
    (identificar_valores_faltantes, generar_reporte_calidad, etc.) nunca
    generan snapshot ni entrada de log (ver record_action) -- no hay nada
    que deshacer de ellas, y no interfieren con esta función.

    NUEVO: el estado actual (antes de deshacer) y el ActionRecord que se
    está deshaciendo se guardan en las pilas de redo, para que
    redo_last_action() pueda reaplicarlos.
    """
    if not session.df_history:
        return False

    snapshot_anterior = session.df_history.pop()
    accion_deshecha = session.actions_log.pop() if session.actions_log else None

    # Guardar el estado ACTUAL (el que se está por reemplazar) en la
    # pila de redo, junto con la acción que le dio origen -- así
    # redo_last_action() puede reaplicar exactamente lo que se deshizo.
    session.redo_df_stack.append(session.current_df)
    if len(session.redo_df_stack) > session.max_undo_steps:
        session.redo_df_stack.pop(0)
    if accion_deshecha is not None:
        session.redo_actions_stack.append(accion_deshecha)

    session.current_df = snapshot_anterior
    return True


def redo_last_action(session: SessionMemory) -> bool:
    """
    NUEVO -- rehace la última acción deshecha con undo_last_action().
    Devuelve True si pudo rehacer, False si no hay nada pendiente de
    rehacer (redo_df_stack vacío -- ya sea porque nunca se deshizo nada,
    o porque se aplicó una acción nueva después del último undo, lo que
    vacía esta pila -- ver record_action).
    """
    if not session.redo_df_stack:
        return False

    snapshot_a_rehacer = session.redo_df_stack.pop()
    accion_a_rehacer = session.redo_actions_stack.pop() if session.redo_actions_stack else None

    # Guardar el estado actual en df_history, igual que haría
    # record_action() con una acción nueva -- para que este redo, a su
    # vez, se pueda volver a deshacer con undo_last_action().
    session.df_history.append(session.current_df)
    if len(session.df_history) > session.max_undo_steps:
        session.df_history.pop(0)
    if accion_a_rehacer is not None:
        session.actions_log.append(accion_a_rehacer)

    session.current_df = snapshot_a_rehacer
    return True


def can_undo(session: SessionMemory) -> bool:
    """NUEVO -- helper para que la UI sepa si conviene habilitar el botón de deshacer."""
    return bool(session.df_history)


def can_redo(session: SessionMemory) -> bool:
    """NUEVO -- helper para que la UI sepa si conviene habilitar el botón de rehacer."""
    return bool(session.redo_df_stack)


def reset_to_original(session: SessionMemory) -> pd.DataFrame:
    """
    Descarta todas las transformaciones aplicadas y vuelve al archivo
    original -- incluye vaciar las pilas de redo (NUEVO: antes no
    existían), porque no tendría sentido poder "rehacer" algo después de
    haber vuelto explícitamente al archivo sin tocar.
    """
    session.current_df = session.original_df.copy()
    session.actions_log.clear()
    session.df_history.clear()
    session.redo_df_stack.clear()
    session.redo_actions_stack.clear()
    return session.current_df


def get_actions_summary(session: SessionMemory) -> list[str]:
    """
    Resumen en lenguaje simple de qué se hizo en la sesión, en orden.
    Pensado para mostrar en Streamlit (ej. un expander "Historial de cambios")
    o para inyectar como contexto adicional al LLM.
    """
    return [
        f"{a.timestamp} · {a.tool_name}({a.arguments}) -> {a.result_summary}"
        for a in session.actions_log
    ]


if __name__ == "__main__":
    # Ejemplo rápido para probar el módulo de forma aislada, simulando
    # una sesión completa con tool calling + compactación + undo/redo.
    df_ejemplo = pd.DataFrame({
        "id": [1, 2, 3],
        "edad": [22, None, 47],
    })

    sesion = create_session(df_ejemplo, system_prompt="Sos un agente de calidad de datos.")
    add_message(sesion, role="user", content="Rellename los nulos de edad con la media")
    add_message(
        sesion, role="assistant", content=None,
        tool_calls=[{"id": "call_1", "type": "function",
                     "function": {"name": "tratar_valores_faltantes", "arguments": "{}"}}],
    )
    add_message(
        sesion, role="tool", content='{"values_filled": {"edad": 1}, "fill_values": {"edad": 34.5}}',
        tool_call_id="call_1", name="tratar_valores_faltantes",
    )
    add_message(sesion, role="assistant", content="Listo, rellené el nulo de edad con el promedio (34.5).")

    print("--- historial ANTES de compactar ---")
    for m in sesion.messages:
        print(" ", m)

    compact_completed_turn(sesion)

    print("\n--- historial DESPUÉS de compactar (mismo contexto, menos tokens) ---")
    for m in sesion.messages:
        print(" ", m)

    print("\n--- mensajes que se le mandan al LLM ---")
    print(get_llm_messages(sesion))

    print("\n--- NUEVO: demo de undo/redo ---")
    df_corregido = df_ejemplo.copy()
    df_corregido["edad"] = df_corregido["edad"].fillna(34.5)
    record_action(
        sesion,
        tool_name="tratar_valores_faltantes",
        arguments={"columna": "edad", "estrategia": "media"},
        result_summary={"values_filled": {"edad": 1}, "fill_values": {"edad": 34.5}},
        new_df=df_corregido,
    )
    print("current_df tras la acción:\n", sesion.current_df)
    print("can_undo:", can_undo(sesion), "| can_redo:", can_redo(sesion))

    undo_last_action(sesion)
    print("\ncurrent_df tras undo:\n", sesion.current_df)
    print("can_undo:", can_undo(sesion), "| can_redo:", can_redo(sesion))

    redo_last_action(sesion)
    print("\ncurrent_df tras redo:\n", sesion.current_df)
    print("can_undo:", can_undo(sesion), "| can_redo:", can_redo(sesion))

    # Acción de solo lectura -- NO debe tocar actions_log ni df_history
    # (ver el FIX en record_action). Se simula pasando new_df=None.
    largo_log_antes = len(sesion.actions_log)
    record_action(
        sesion,
        tool_name="generar_reporte_calidad",
        arguments={},
        result_summary={"quality_score": 85},
        new_df=None,
    )
    assert len(sesion.actions_log) == largo_log_antes, "una acción de solo lectura NO debe loguearse"
    print("\nOK: la acción de solo lectura no modificó actions_log.")