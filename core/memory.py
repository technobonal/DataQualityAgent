"""
core/memory.py

Memoria de sesión del agente: guarda tres cosas a la vez, todas atadas a
la misma sesión de un usuario trabajando sobre un archivo:

  1. El historial de la conversación (mensajes en formato OpenAI/Groq,
     para que llm_client.py tenga contexto entre turnos).
  2. El estado actual del DataFrame (qué transformaciones ya se aplicaron).
  3. Un log de acciones (qué tool se llamó, con qué argumentos y qué
     resultado dio) — sirve como auditoría y para poder deshacer pasos.

Sin dependencia de LLM en sí misma — es estado puro, testeable en
aislado. llm_client.py y agent.py son quienes la usan.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

import pandas as pd

# Cuántos mensajes de historial (sin contar el system prompt) se le mandan
# al LLM como máximo. Evita que la conversación crezca sin límite y se
# coma el context window / la cuota de tokens.
_MAX_HISTORY_MESSAGES = 20

# Cuántos snapshots del DataFrame se guardan para poder deshacer pasos.
_MAX_UNDO_STEPS = 10


@dataclass
class ActionRecord:
    """Registro de una acción (tool) ya aplicada al dataset en esta sesión."""
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
    df_history:   snapshots de current_df ANTES de cada acción, para poder
                  deshacer. Limitado a _MAX_UNDO_STEPS para no consumir
                  memoria de más en datasets grandes.
    """
    original_df: pd.DataFrame
    current_df: pd.DataFrame
    messages: list[dict[str, Any]] = field(default_factory=list)
    actions_log: list[ActionRecord] = field(default_factory=list)
    df_history: list[pd.DataFrame] = field(default_factory=list)
    max_history_messages: int = _MAX_HISTORY_MESSAGES
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


def get_llm_messages(session: SessionMemory) -> list[dict[str, Any]]:
    """
    Devuelve los mensajes a enviar al LLM, recortando el historial si es
    muy largo. Si el primer mensaje es el system prompt, se conserva
    siempre (no cuenta contra el límite) y se recortan solo los mensajes
    de conversación más viejos.
    """
    if not session.messages:
        return []

    if session.messages[0].get("role") == "system":
        system_msg, resto = session.messages[0], session.messages[1:]
        if len(resto) > session.max_history_messages:
            resto = resto[-session.max_history_messages:]
        return [system_msg] + resto

    if len(session.messages) > session.max_history_messages:
        return session.messages[-session.max_history_messages:]
    return list(session.messages)


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
    """
    if new_df is not None:
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


def undo_last_action(session: SessionMemory) -> bool:
    """
    Deshace el último paso que modificó el dataset (vuelve current_df al
    snapshot anterior). Devuelve True si pudo deshacer, False si no hay
    nada para deshacer.

    Nota: solo deshace cambios al DataFrame. Las tools de "solo lectura"
    (identificar_valores_faltantes, generar_reporte_calidad, etc.) no
    generan snapshot porque no modifican el dataset.
    """
    if not session.df_history:
        return False

    session.current_df = session.df_history.pop()
    if session.actions_log:
        session.actions_log.pop()
    return True


def reset_to_original(session: SessionMemory) -> pd.DataFrame:
    """Descarta todas las transformaciones aplicadas y vuelve al archivo original."""
    session.current_df = session.original_df.copy()
    session.actions_log.clear()
    session.df_history.clear()
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
    # una sesión completa: crear, chatear, aplicar tools, deshacer.
    df_ejemplo = pd.DataFrame({
        "id": [1, 2, 3],
        "edad": [22, None, 47],
    })

    sesion = create_session(df_ejemplo, system_prompt="Sos un agente de calidad de datos.")
    add_message(sesion, role="user", content="Rellename los nulos de edad con la media")

    print("--- mensajes para el LLM ---")
    print(get_llm_messages(sesion))

    # Simulamos que el LLM decidió llamar a tratar_valores_faltantes
    df_nuevo = df_ejemplo.copy()
    df_nuevo["edad"] = df_nuevo["edad"].fillna(df_nuevo["edad"].mean())
    record_action(
        sesion,
        tool_name="tratar_valores_faltantes",
        arguments={"strategy": "mean", "columns": ["edad"]},
        result_summary={"values_filled": {"edad": 1}},
        new_df=df_nuevo,
    )

    print("\n--- dataset actual (después de la acción) ---")
    print(sesion.current_df)

    print("\n--- resumen de acciones ---")
    for linea in get_actions_summary(sesion):
        print(" ", linea)

    print("\n--- undo_last_action() ---")
    deshecho = undo_last_action(sesion)
    print("¿Se pudo deshacer?", deshecho)
    print(sesion.current_df)
    print("Acciones restantes:", len(sesion.actions_log))