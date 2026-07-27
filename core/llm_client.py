"""
core/llm_client.py

Cliente unificado del LLM: encadena varios modelos de Groq (cada uno con
su propia cuota de tokens/minuto separada) y, si TODOS fallan, cae como
último recurso a Ollama local — todo a través de litellm.

INTEGRACIÓN CON RATE LIMITER: chat_completion() ahora recibe el
RateLimiter de la SESIÓN actual (session.rate_limiter, en
core/memory.py) como parámetro -- no uno global del módulo. Así cada
sesión de Streamlit lleva su propio cupo de tokens/minuto, sin
compartirlo con otras sesiones que puedan estar corriendo al mismo
tiempo. Antes de intentar la cadena de Groq, se le pregunta a ese
limiter cuál es el primer modelo que todavía tiene cupo disponible, y
se arranca por ahí -- en vez de siempre empezar por
llama-3.3-70b-versatile y descubrir recién con un 429 que ya no tenía
cupo.
"""

from __future__ import annotations

import json
import os
from typing import Any

import litellm
from dotenv import load_dotenv

from security.rate_limiter import RateLimiter, estimar_tokens

load_dotenv()

USE_LOCAL = os.getenv("USE_LOCAL", "false").strip().lower() == "true"

# --- Ollama local ---
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

_DEFAULT_GROQ_CHAIN = "llama-3.3-70b-versatile,openai/gpt-oss-20b,openai/gpt-oss-120b,llama-3.1-8b-instant"
GROQ_MODEL_CHAIN = [
    m.strip() for m in os.getenv("GROQ_MODEL_CHAIN", _DEFAULT_GROQ_CHAIN).split(",") if m.strip()
]


class LLMClientError(Exception):
    """Se lanza cuando falta la API key de Groq, o cuando Groq y el fallback fallan."""


def _call_groq(model: str, messages, tools, tool_choice, temperature, max_tokens):
    """Una sola llamada a un modelo de Groq. Sin api_base -> usa el endpoint real de Groq."""
    kwargs: dict[str, Any] = {
        "model": f"groq/{model}",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "num_retries": 2,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    return litellm.completion(**kwargs)


def _call_ollama(messages, tools, tool_choice, temperature, max_tokens):
    """Una sola llamada a Ollama local. Acá SÍ corresponde api_base."""
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
    return litellm.completion(**kwargs)


def chat_completion(
    messages: list[dict[str, Any]],
    rate_limiter: RateLimiter,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str = "auto",
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> Any:
    """
    rate_limiter: el de la SESIÓN actual -- pasar session.rate_limiter
                  (ver core/memory.py). Obligatorio: así cada sesión
                  lleva su propio cupo, sin compartirlo con otras.

    Con USE_LOCAL=true: Ollama local directo (sin tocar Groq, sin
    consultar el rate limiter -- no aplica).

    Con USE_LOCAL=false (default):
      1. Le pregunta a rate_limiter cuál es el primer modelo de
         GROQ_MODEL_CHAIN que todavía tiene cupo de tokens/minuto.
      2. Si NINGÚN modelo tiene cupo en este momento, salta directo a
         Ollama local (evita mandarle a Groq una llamada que ya
         sabemos que va a rechazar).
      3. Si hay un modelo con cupo, arranca la cadena por ESE modelo
         (reordenada), y sigue con el resto como respaldo si falla por
         otro motivo (no por cuota). Cada llamada exitosa registra su
         gasto real en rate_limiter.
      4. Si toda la cadena de Groq falla igual, cae a Ollama como red
         de seguridad final.
    """
    if USE_LOCAL:
        try:
            return _call_ollama(messages, tools, tool_choice, temperature, max_tokens)
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

    tokens_estimados = estimar_tokens(messages, tools)
    modelo_sugerido = rate_limiter.siguiente_modelo_disponible(GROQ_MODEL_CHAIN, tokens_estimados)

    if modelo_sugerido is None:
        # Toda la cadena de Groq está sin cupo ahora mismo -> directo a Ollama,
        # sin gastar una llamada de red que ya sabemos que Groq va a rechazar.
        try:
            return _call_ollama(messages, tools, tool_choice, temperature, max_tokens)
        except Exception as e:
            raise LLMClientError(
                f"Los {len(GROQ_MODEL_CHAIN)} modelos de Groq están sin cupo de tokens/minuto "
                f"en este momento, y Ollama local también falló. Detalle de Ollama: {e}"
            ) from e

    # Reordena la cadena para arrancar por el modelo con cupo disponible;
    # el resto queda como respaldo si ese falla por otro motivo (no por cuota).
    orden = [modelo_sugerido] + [m for m in GROQ_MODEL_CHAIN if m != modelo_sugerido]

    ultimo_error: Exception | None = None

    for modelo in orden:
        try:
            respuesta = _call_groq(modelo, messages, tools, tool_choice, temperature, max_tokens)
            rate_limiter.registrar_uso(modelo, tokens_estimados)  # registra el gasto real, solo si tuvo éxito
            return respuesta
        except litellm.BadRequestError as e:
            if "tool_use_failed" in str(e):
                try:
                    respuesta = _call_groq(modelo, messages, tools, tool_choice, temperature, max_tokens)
                    rate_limiter.registrar_uso(modelo, tokens_estimados)
                    return respuesta
                except Exception as e2:
                    ultimo_error = e2
                    continue
            ultimo_error = e
            continue
        except Exception as e:
            ultimo_error = e
            continue

    try:
        return _call_ollama(messages, tools, tool_choice, temperature, max_tokens)
    except Exception as e_ollama:
        raise LLMClientError(
            f"Falló la llamada al LLM en toda la cadena ({len(GROQ_MODEL_CHAIN)} modelos de "
            f"Groq + Ollama local). Último error de Groq: {ultimo_error}. "
            f"Error de Ollama: {e_ollama}"
        ) from e_ollama


def extract_tool_calls(response: Any) -> list[dict[str, Any]]:
    """
    Extrae las tool calls de una respuesta del LLM en formato simple.
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
    """Extrae el texto de la respuesta (None si el modelo solo hizo tool calls)."""
    return response.choices[0].message.content


def message_to_dict(response: Any) -> dict[str, Any]:
    """
    Convierte el mensaje de la respuesta a un dict plano en formato
    OpenAI/Groq, listo para guardar en el historial de core/memory.py.
    """
    mensaje = response.choices[0].message

    if hasattr(mensaje, "model_dump"):
        return mensaje.model_dump()

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
    from security.rate_limiter import RateLimiter as _RL
    mensajes = [
        {"role": "system", "content": "Sos un agente de calidad de datos, respondé breve."},
        {"role": "user", "content": "Decime 'hola' en una sola palabra."},
    ]
    respuesta = chat_completion(mensajes, rate_limiter=_RL())
    print(extract_text(respuesta))