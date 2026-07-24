"""
core/llm_client.py

Cliente unificado del LLM: encadena varios modelos de Groq (cada uno con
su propia cuota de tokens/minuto separada — ver security/rate_limiter.py
y la conversación del proyecto) y, si TODOS fallan, cae como último
recurso a Ollama local — todo a través de litellm, que expone Groq y
Ollama con la misma interfaz (formato OpenAI).

Cadena de modelos (orden real, confirmado con la cuenta free de Groq):
  1. llama-3.3-70b-versatile  (12K TPM — mejor calidad de razonamiento)
  2. openai/gpt-oss-20b       (8K TPM  — el más rápido, ~1000 t/seg)
  3. openai/gpt-oss-120b      (8K TPM  — más capaz que el 20b)
  4. llama-3.1-8b-instant     (6K TPM, pero 14.4K RPD — el mayor margen diario)
  5. ollama_chat/<modelo>     (red final, sin límite externo, pero lento)

USE_LOCAL=true en el .env: se salta toda la cadena de Groq y usa
Ollama DIRECTO — pensado para desarrollar/probar sin gastar ninguna
cuota de Groq (aunque sea más lento). USE_LOCAL=false (default): usa la
cadena de Groq, con Ollama como red de seguridad automática si los 4
modelos de Groq fallan.

Sin lógica de negocio acá — solo la conexión con el LLM. Quién decide
qué tool llamar según la respuesta vive en core/agent.py. La memoria de
la conversación vive en core/memory.py.
"""

from __future__ import annotations

import json
import os
from typing import Any

import litellm
from dotenv import load_dotenv

load_dotenv()

USE_LOCAL = os.getenv("USE_LOCAL", "false").strip().lower() == "true"

# --- Ollama local (siempre disponible como red final; único proveedor si USE_LOCAL=true) ---
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Cadena de modelos de Groq, en orden de preferencia. Configurable vía
# .env (GROQ_MODEL_CHAIN, separados por coma) si en algún momento cambia
# el catálogo o los límites de tu cuenta — ver
# https://console.groq.com/docs/models para el catálogo vigente.
_DEFAULT_GROQ_CHAIN = "llama-3.3-70b-versatile,openai/gpt-oss-20b,openai/gpt-oss-120b,llama-3.1-8b-instant"
GROQ_MODEL_CHAIN = [
    m.strip() for m in os.getenv("GROQ_MODEL_CHAIN", _DEFAULT_GROQ_CHAIN).split(",") if m.strip()
]


class LLMClientError(Exception):
    """Se lanza cuando falta la API key de Groq, o cuando Groq y el fallback fallan."""


def chat_completion(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str = "auto",
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> Any:
    """
    Llama al LLM.

    Con USE_LOCAL=true: usa Ollama local DIRECTO, sin pasar por Groq —
    pensado para desarrollar/probar sin gastar ninguna cuota.

    Con USE_LOCAL=false (default): intenta la cadena de GROQ_MODEL_CHAIN
    en orden (cada modelo tiene su propia cuota de tokens/minuto
    separada en Groq, así que encadenarlos multiplica el margen
    disponible antes de toparse con un límite real). Si TODOS los
    modelos de Groq fallan, cae automáticamente a Ollama local como red
    de seguridad final — litellm maneja este encadenado solo, vía el
    parámetro 'fallbacks'.

    messages: historial en formato OpenAI/Groq (ver core/memory.py ->
              get_llm_messages()).
    tools:    TOOLS_SCHEMA de tools/tool_definitions.py, para habilitar
              function/tool calling. None para chat libre sin tools.
    tool_choice: 'auto' deja que el modelo decida si llama una tool o
              responde directo en texto. Se ignora si tools es None.

    Devuelve el objeto de respuesta de litellm (mismo formato que la API
    de OpenAI) — usar extract_text() / extract_tool_calls() para leerlo
    sin acoplarse a la estructura interna de litellm.

    Lanza LLMClientError si falta la config necesaria, o si toda la
    cadena (Groq completo + Ollama final) falla.
    """
    if USE_LOCAL:
        kwargs: dict[str, Any] = {
            "model": f"ollama_chat/{OLLAMA_MODEL}",
            "api_base": OLLAMA_API_BASE,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        try:
            return litellm.completion(**kwargs)
        except Exception as e:
            raise LLMClientError(
                f"Falló la llamada a Ollama local ({OLLAMA_MODEL} en {OLLAMA_API_BASE}). "
                f"¿Está 'ollama serve' corriendo y el modelo descargado "
                f"(ollama pull {OLLAMA_MODEL})? Detalle: {e}"
            ) from e

    if not GROQ_API_KEY:
        raise LLMClientError(
            "Falta GROQ_API_KEY en el archivo .env. Conseguila gratis en "
            "https://console.groq.com/keys"
        )

    modelo_primario = f"groq/{GROQ_MODEL_CHAIN[0]}"
    # El resto de la cadena de Groq, más Ollama al final como red de
    # seguridad absoluta (siempre disponible, sin límite externo).
    cadena_fallback = [f"groq/{m}" for m in GROQ_MODEL_CHAIN[1:]]
    cadena_fallback.append(f"ollama_chat/{OLLAMA_MODEL}")

    kwargs = {
        "model": modelo_primario,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "num_retries": 2,
        "fallbacks": cadena_fallback,
        "api_base": OLLAMA_API_BASE,  # solo lo usa litellm si el fallback termina siendo ollama_chat/
    }

    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice

    try:
        return litellm.completion(**kwargs)
    except litellm.RateLimitError as e:
        raise LLMClientError(
            f"Los {len(GROQ_MODEL_CHAIN)} modelos de Groq y Ollama local fallaron por límite de "
            f"tokens/cuota. Esperá unos segundos y volvé a intentar. Detalle técnico: {e}"
        ) from e
    except litellm.BadRequestError as e:
        # Groq a veces genera una tool call mal formada (formato XML-like
        # en vez de JSON válido, típico de function-calling estructurado
        # bajo carga) — es transitorio en la mayoría de los casos, así que
        # vale la pena un segundo intento antes de darse por vencido.
        if "tool_use_failed" in str(e):
            try:
                return litellm.completion(**kwargs)
            except Exception as e2:
                raise LLMClientError(
                    f"El modelo no pudo generar una llamada a tool válida, ni al reintentar: {e2}"
                ) from e2
        raise LLMClientError(f"Falló la llamada al LLM: {e}") from e
    except Exception as e:
        raise LLMClientError(
            f"Falló la llamada al LLM en toda la cadena ({len(GROQ_MODEL_CHAIN)} modelos de "
            f"Groq + Ollama local): {e}"
        ) from e


def extract_tool_calls(response: Any) -> list[dict[str, Any]]:
    """
    Extrae las tool calls de una respuesta del LLM, en un formato simple
    y desacoplado de la estructura interna de litellm — lo que consume
    core/agent.py para saber qué ejecutar.

    Devuelve una lista de {"id": ..., "name": ..., "arguments": dict}.
    Lista vacía si el modelo respondió directo en texto, sin usar tools.
    """
    mensaje = response.choices[0].message
    tool_calls = getattr(mensaje, "tool_calls", None) or []

    resultado = []
    for tc in tool_calls:
        try:
            arguments = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, TypeError):
            arguments = {}
        resultado.append({
            "id": tc.id,
            "name": tc.function.name,
            "arguments": arguments,
        })
    return resultado


def extract_text(response: Any) -> str | None:
    """Extrae el texto de la respuesta (None si el modelo solo hizo tool calls, sin texto)."""
    return response.choices[0].message.content


def message_to_dict(response: Any) -> dict[str, Any]:
    """
    Convierte el mensaje de la respuesta del LLM a un dict plano en
    formato OpenAI/Groq, listo para guardarlo tal cual en el historial
    de core/memory.py — necesario porque cuando el modelo pide tool
    calls, el próximo mensaje que se le mande de vuelta tiene que incluir
    ese mismo assistant message con sus tool_calls originales (con los
    argumentos todavía como string JSON, sin parsear), o Groq rechaza la
    conversación por formato inválido.
    """
    mensaje = response.choices[0].message

    if hasattr(mensaje, "model_dump"):
        return mensaje.model_dump()

    # Fallback manual, usado en los tests con objetos simulados (donde
    # no hay model_dump real de pydantic, como sí lo tiene litellm).
    tool_calls_raw = getattr(mensaje, "tool_calls", None)
    tool_calls = None
    if tool_calls_raw:
        tool_calls = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in tool_calls_raw
        ]
    return {"role": "assistant", "content": mensaje.content, "tool_calls": tool_calls}


if __name__ == "__main__":
    # Prueba manual — requiere GROQ_API_KEY real en el .env.
    # IMPORTANTE: este bloque no se puede correr desde el sandbox de
    # Claude (sin acceso a api.groq.com) — la lógica de parseo ya se probó
    # con respuestas simuladas (ver conversación), pero la llamada real a
    # Groq hay que confirmarla desde tu máquina.
    mensajes = [
        {"role": "system", "content": "Sos un agente de calidad de datos, respondé breve."},
        {"role": "user", "content": "Decime 'hola' en una sola palabra."},
    ]
    respuesta = chat_completion(mensajes)
    print(extract_text(respuesta))