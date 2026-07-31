"""
core/llm_client.py

Cliente unificado del LLM: encadena varios modelos de Groq (cada uno con
su propia cuota de tokens/minuto separada) y, si TODOS fallan, cae a
Ollama Cloud (Minimax, nivel intermedio) y, si eso también falla, cae
como último recurso a Ollama local — todo a través de litellm.

Cadena completa: Groq (4 modelos) -> Ollama Cloud (minimax-m2.7:cloud)
-> Ollama local (qwen2.5:7b)

NUEVO: predecir_proveedor() -- antes de llamar a chat_completion(), le
pregunta al mismo rate_limiter qué proveedor se va a usar (Groq, Ollama
Cloud u Ollama local), SIN hacer la llamada real. Esto permite avisarle
al usuario ANTES de que empiece la espera larga (si va a caer a Ollama
local), en vez de que el spinner diga siempre el mismo mensaje genérico
y el usuario piense que la app se colgó quedándose sin respuesta -- como
pasó hoy con la cadena de Groq sin cupo cayendo a qwen2.5:7b sin aviso.

También se agregó logging explícito (visible en la terminal) de qué
proveedor/modelo respondió realmente cada llamada, más allá de los
logs genéricos que ya imprime litellm.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import litellm
from dotenv import load_dotenv

from security.rate_limiter import RateLimiter, estimar_tokens

load_dotenv()

logger = logging.getLogger("dataqualityagent.llm_client")

USE_LOCAL = os.getenv("USE_LOCAL", "false").strip().lower() == "true"

# --- Ollama local (último nivel de la cadena) ---
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

# --- Ollama Cloud (nivel intermedio, entre Groq y Ollama local) ---
OLLAMA_CLOUD_API_KEY = os.getenv("OLLAMA_CLOUD_API_KEY", "")
OLLAMA_CLOUD_MODEL = os.getenv("OLLAMA_CLOUD_MODEL", "minimax-m2.7:cloud")
OLLAMA_CLOUD_BASE = os.getenv("OLLAMA_CLOUD_BASE", "https://ollama.com/v1")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

_DEFAULT_GROQ_CHAIN = "llama-3.3-70b-versatile,openai/gpt-oss-20b,openai/gpt-oss-120b,llama-3.1-8b-instant"
GROQ_MODEL_CHAIN = [
    m.strip() for m in os.getenv("GROQ_MODEL_CHAIN", _DEFAULT_GROQ_CHAIN).split(",") if m.strip()
]


class LLMClientError(Exception):
    """Se lanza cuando falta la API key de Groq, o cuando Groq, Ollama Cloud y el fallback local fallan."""


def predecir_proveedor(
    messages: list[dict[str, Any]],
    rate_limiter: RateLimiter,
    tools: list[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    """
    Predice qué proveedor va a usar la PRÓXIMA llamada a chat_completion,
    SIN hacer ninguna llamada real -- mismo cálculo que chat_completion
    hace internamente (rate_limiter.siguiente_modelo_disponible), pero
    disponible ANTES, para poder avisarle al usuario que se viene una
    espera larga (Ollama Cloud u Ollama local) antes de que arranque.

    Devuelve (proveedor, detalle):
      ("local", OLLAMA_MODEL)                  si USE_LOCAL=true
      ("groq", nombre_del_modelo)              si hay cupo en algún modelo de Groq
      ("cloud_fallback", OLLAMA_CLOUD_MODEL)    si Groq está configurado pero
                                                 NINGÚN modelo de la cadena
                                                 tiene cupo ahora mismo
                                                 (se intenta Ollama Cloud antes
                                                 que el local)
    """
    if USE_LOCAL:
        return "local", OLLAMA_MODEL

    if not GROQ_API_KEY:
        return "cloud_fallback", OLLAMA_CLOUD_MODEL

    tokens_estimados = estimar_tokens(messages, tools)
    modelo = rate_limiter.siguiente_modelo_disponible(GROQ_MODEL_CHAIN, tokens_estimados)

    if modelo is None:
        return "cloud_fallback", OLLAMA_CLOUD_MODEL
    return "groq", modelo


def _call_groq(model: str, messages, tools, tool_choice, temperature, max_tokens):
    """Una sola llamada a un modelo de Groq. Sin api_base -> usa el endpoint real de Groq."""
    kwargs: dict[str, Any] = {
        "model": f"groq/{model}",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "num_retries": 0,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    return litellm.completion(**kwargs)


def _call_ollama_cloud(messages, tools, tool_choice, temperature, max_tokens):
    """
    Una sola llamada a Ollama Cloud (nivel intermedio). A diferencia de
    Ollama local, este endpoint SÍ requiere api_key (es compatible con
    OpenAI, por eso el prefijo "openai/" para litellm en vez de
    "ollama_chat/").
    """
    kwargs: dict[str, Any] = {
        "model": f"openai/{OLLAMA_CLOUD_MODEL}",
        "api_base": OLLAMA_CLOUD_BASE,
        "api_key": OLLAMA_CLOUD_API_KEY,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    return litellm.completion(**kwargs)


def _call_ollama(messages, tools, tool_choice, temperature, max_tokens):
    """Una sola llamada a Ollama local. Acá SÍ corresponde api_base, sin api_key."""
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

    Con USE_LOCAL=true: Ollama local directo (sin tocar Groq ni Cloud,
    sin consultar el rate limiter -- no aplica).

    Con USE_LOCAL=false (default), cadena de 3 niveles:
      1. Le pregunta a rate_limiter cuál es el primer modelo de
         GROQ_MODEL_CHAIN que todavía tiene cupo de tokens/minuto.
      2. Si NINGÚN modelo tiene cupo en este momento, salta directo a
         Ollama Cloud (evita mandarle a Groq una llamada que ya
         sabemos que va a rechazar).
      3. Si hay un modelo con cupo, arranca la cadena por ESE modelo
         (reordenada), y sigue con el resto como respaldo si falla por
         otro motivo (no por cuota). Cada llamada exitosa registra su
         gasto real en rate_limiter.
      4. Si toda la cadena de Groq falla igual (por error, no por
         cuota agotada), se intenta Ollama Cloud.
      5. Si Ollama Cloud también falla (o no tiene API key
         configurada), cae a Ollama local como red de seguridad final.
    """
    # --- NUEVO: Limpieza de seguridad del historial AQUÍ ---
    # Lo limpiamos antes de mandarlo a cualquier proveedor y antes de contar tokens
    mensajes_limpios = []
    for msg in messages:
        msg_limpio = {k: v for k, v in msg.items() if k != "reasoning"}
        mensajes_limpios.append(msg_limpio)
    messages = mensajes_limpios
    # -------------------------------------------------------

    if USE_LOCAL:
        logger.info(f"[chat_completion] Modo local forzado -> ollama_chat/{OLLAMA_MODEL}")
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

    error_groq: Exception | None = None

    if modelo_sugerido is None:
        logger.info(
            "[chat_completion] Los 4 modelos de Groq están sin cupo de tokens/minuto -> "
            f"probando Ollama Cloud (openai/{OLLAMA_CLOUD_MODEL})."
        )
        orden = []
    else:
        # Reordena la cadena para arrancar por el modelo con cupo disponible;
        # el resto queda como respaldo si ese falla por otro motivo (no por cuota).
        orden = [modelo_sugerido] + [m for m in GROQ_MODEL_CHAIN if m != modelo_sugerido]

    # --- INICIO DEL BUCLE GROQ MODIFICADO ---
    for modelo in orden:
        rate_limiter.registrar_request(modelo)  # cuenta el intento, éxito o no
        try:
            respuesta = _call_groq(modelo, messages, tools, tool_choice, temperature, max_tokens)
            rate_limiter.registrar_uso(modelo, tokens_estimados)
            logger.info(f"[chat_completion] Respondió groq/{modelo}")
            return respuesta
            
        except litellm.BadRequestError as e:
            if "tool_use_failed" in str(e):
                try:
                    rate_limiter.registrar_request(modelo)
                    respuesta = _call_groq(modelo, messages, tools, tool_choice, temperature, max_tokens)
                    rate_limiter.registrar_uso(modelo, tokens_estimados)
                    logger.info(f"[chat_completion] Respondió groq/{modelo} (tras reintento)")
                    return respuesta
                except Exception as e2:
                    error_groq = e2
                    # NUEVO LOG: Muestra por qué falló el reintento
                    logger.warning(f"[chat_completion] Reintento fallido con groq/{modelo}: {str(e2)}")
                    continue
                    
            error_groq = e
            # NUEVO LOG: Muestra por qué falló con BadRequest
            logger.warning(f"[chat_completion] Falló groq/{modelo} (BadRequest): {str(e)}")
            continue
            
        except Exception as e:
            error_groq = e
            # NUEVO LOG: Muestra cualquier otro error general
            logger.warning(f"[chat_completion] Falló groq/{modelo}: {str(e)}")
            continue  
    # --- FIN DEL BUCLE GROQ MODIFICADO ---

    if orden:
        logger.info(
            "[chat_completion] Toda la cadena de Groq falló -> "
            f"probando Ollama Cloud (openai/{OLLAMA_CLOUD_MODEL})."
        )

    # --- Nivel intermedio: Ollama Cloud ---
    error_cloud: Exception | None = None
    if OLLAMA_CLOUD_API_KEY:
        try:
            respuesta = _call_ollama_cloud(messages, tools, tool_choice, temperature, max_tokens)
            logger.info(f"[chat_completion] Respondió openai/{OLLAMA_CLOUD_MODEL} (Ollama Cloud)")
            return respuesta
        except Exception as e:
            error_cloud = e
            # Añadimos también el detalle del error en el log de Cloud
            logger.info(
                f"[chat_completion] Falló Ollama Cloud ({OLLAMA_CLOUD_MODEL}) Detalle: {str(e)} -> "
                f"cayendo a Ollama local (ollama_chat/{OLLAMA_MODEL}), va a tardar más."
            )
    else:
        error_cloud = LLMClientError("Falta OLLAMA_CLOUD_API_KEY en el .env, se saltea este nivel.")
        logger.info(
            "[chat_completion] OLLAMA_CLOUD_API_KEY no configurada -> "
            f"cayendo directo a Ollama local (ollama_chat/{OLLAMA_MODEL})."
        )

    # --- Último nivel: Ollama local ---
    try:
        respuesta = _call_ollama(messages, tools, tool_choice, temperature, max_tokens)
        logger.info(f"[chat_completion] Respondió ollama_chat/{OLLAMA_MODEL} (fallback final)")
        return respuesta
    except Exception as e_ollama:
        raise LLMClientError(
            f"Falló la llamada al LLM en toda la cadena ({len(GROQ_MODEL_CHAIN)} modelos de "
            f"Groq + Ollama Cloud + Ollama local). "
            f"Último error de Groq: {error_groq}. "
            f"Error de Ollama Cloud: {error_cloud}. "
            f"Error de Ollama local: {e_ollama}"
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
    limiter = _RL()

    print("--- predecir_proveedor con presupuesto lleno ---")
    mensajes = [{"role": "user", "content": "hola"}]
    proveedor, detalle = predecir_proveedor(mensajes, limiter)
    print(f"proveedor={proveedor}  detalle={detalle}")

    print("\n--- predecir_proveedor tras agotar los 4 modelos de Groq ---")
    from security.rate_limiter import MODEL_TPM_LIMITS
    for modelo, limite in MODEL_TPM_LIMITS.items():
        limiter.registrar_uso(modelo, int(limite * 0.85))  # supera el margen de seguridad del 80%
    proveedor, detalle = predecir_proveedor(mensajes, limiter)
    print(f"proveedor={proveedor}  detalle={detalle}  (esperado: cloud_fallback)")