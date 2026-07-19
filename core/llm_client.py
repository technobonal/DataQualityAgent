"""
core/llm_client.py

Cliente unificado del LLM: habla con Groq como proveedor principal, y si
falla (cuota agotada, error de servicio, timeout), reintenta
automáticamente con un modelo de Hugging Face Inference Providers —
todo a través de litellm, que expone ambos proveedores con la misma
interfaz (formato OpenAI).

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

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

# Modelo de respaldo en Hugging Face Inference Providers, formato litellm:
#   huggingface/<provider>/<hf_org_or_user>/<hf_model>
# 'hf-inference' es el proveedor serverless propio de Hugging Face (no
# requiere elegir un tercero como Fireworks/Together/Novita).
# El catálogo de modelos disponibles cambia seguido — si este deja de
# funcionar, revisar el catálogo vigente en:
# https://huggingface.co/docs/inference-providers
HF_FALLBACK_MODEL = os.getenv(
    "HF_FALLBACK_MODEL",
    "huggingface/hf-inference/meta-llama/Llama-3.3-70B-Instruct",
)

# litellm espera el token de Hugging Face en la variable HF_TOKEN, pero en
# nuestro .env lo llamamos HUGGINGFACE_API_KEY (más claro para el resto
# del proyecto) -> lo adaptamos acá para no duplicar la variable.
if HUGGINGFACE_API_KEY and not os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = HUGGINGFACE_API_KEY


class LLMClientError(Exception):
    """Se lanza cuando falta la API key de Groq, o cuando Groq y el fallback fallan."""


def chat_completion(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str = "auto",
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> Any:
    """
    Llama al LLM con Groq como proveedor principal y, si HUGGINGFACE_API_KEY
    está configurada, fallback automático a Hugging Face si Groq falla
    (cuota agotada, error 5xx, timeout, etc. — litellm decide cuándo
    reintentar con el fallback según el tipo de error).

    messages: historial en formato OpenAI/Groq (ver core/memory.py ->
              get_llm_messages()).
    tools:    TOOLS_SCHEMA de tools/tool_definitions.py, para habilitar
              function/tool calling. None para chat libre sin tools.
    tool_choice: 'auto' deja que el modelo decida si llama una tool o
              responde directo en texto. Se ignora si tools es None.

    Devuelve el objeto de respuesta de litellm (mismo formato que la API
    de OpenAI) — usar extract_text() / extract_tool_calls() para leerlo
    sin acoplarse a la estructura interna de litellm.

    Lanza LLMClientError si falta la API key de Groq, o si tanto Groq
    como el fallback (si está configurado) fallan.
    """
    if not GROQ_API_KEY:
        raise LLMClientError(
            "Falta GROQ_API_KEY en el archivo .env. Conseguila gratis en "
            "https://console.groq.com/keys"
        )

    kwargs: dict[str, Any] = {
        "model": f"groq/{GROQ_MODEL}",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "num_retries": 2,
    }

    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice

    if HUGGINGFACE_API_KEY:
        kwargs["fallbacks"] = [HF_FALLBACK_MODEL]

    try:
        return litellm.completion(**kwargs)
    except Exception as e:
        detalle_fallback = " (con fallback a Hugging Face incluido)" if HUGGINGFACE_API_KEY else ""
        raise LLMClientError(f"Falló la llamada al LLM{detalle_fallback}: {e}") from e


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