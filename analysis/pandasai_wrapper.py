"""
analysis/pandasai_wrapper.py

Análisis conversacional libre sobre el dataset (a diferencia de las 15
tools fijas y deterministas de core/agent.py, acá el LLM SÍ genera
código pandas arbitrario para responder preguntas abiertas).

DECISIÓN DE DISEÑO (cambió tras revisar security/sandbox.py): no se usa
el executor interno de la librería PandasAI, que ejecuta el código él
mismo -- sin pasar por nuestra sandbox -- salvo que se le pase un
`sandbox=DockerSandbox()`, que requiere Docker corriendo. Eso no encaja
ni con la instancia Oracle de 1 OCPU (peso extra de más) ni con el
sandbox que ya construiste (proceso aislado + builtins restringidos,
sin Docker).

En cambio, acá se arma un prompt propio -- liviano, sin mandar el
dataset completo, solo metadata (columnas, dtypes, head) -- se le pide
al LLM código que asigne su resultado a `resultado` (la misma
convención que ya espera security/sandbox.py), y la EJECUCIÓN corre
siempre a través de run_sandboxed(). El LLM nunca ejecuta nada por su
cuenta.

Reutiliza la MISMA cadena de fallback de core/llm_client.py (Groq ->
Ollama Cloud -> Ollama local) y el rate_limiter de la sesión -- el
análisis libre consume del mismo cupo que el chat principal, no de uno
aparte.

Si en algún momento se necesita explícitamente el ecosistema completo
de PandasAI (gráficos con su propio prompt template, multi-dataframe,
etc.), se puede instalar `pandasai` y usarlo solo para GENERAR el
código (vía su LLM base + generate_code()), pero la ejecución debería
seguir pasando por run_sandboxed() en vez de su executor interno, por
la misma razón de arriba.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from core.llm_client import chat_completion, extract_text, LLMClientError
from security.rate_limiter import RateLimiter
from security.sandbox import run_sandboxed, SandboxResult

# Cuántas filas de muestra (df.head) se incluyen en el prompt -- alcanza
# para que el LLM entienda la forma de los datos (columnas, tipos,
# formato de los valores) sin mandar el dataset completo. El código que
# genere corre sobre el DataFrame COMPLETO en la sandbox, esta muestra
# es solo para que el LLM sepa qué código escribir.
_FILAS_DE_MUESTRA = 5

# Si el código generado falla en la sandbox por un error real de
# ejecución (no por rechazo del análisis estático), cuántas veces se le
# da al LLM la chance de corregirlo mostrándole el error. Un rechazo por
# análisis estático NUNCA se reintenta -- ver analizar_con_pandasai.
_MAX_REINTENTOS_POR_ERROR = 1

_PROMPT_BASE = """\
Sos un asistente que traduce preguntas en lenguaje natural sobre un \
dataset a código Python (pandas). NO respondas la pregunta en texto -- \
únicamente devolvé el código.

Información del dataset (ya cargado en la variable `df`):
- Columnas y tipos: {dtypes}
- Primeras {n_filas} filas de muestra:
{muestra}

Reglas estrictas:
- El DataFrame ya existe en una variable llamada `df` -- no lo vuelvas \
a crear ni le pidas al usuario que lo suba.
- Tu código DEBE asignar el resultado final a una variable llamada \
`resultado`. Ej.: resultado = df['edad'].mean()
- Está PROHIBIDO importar cualquier módulo (os, sys, subprocess, \
requests, etc.) -- solo pandas (ya disponible como `pd`) y las \
funciones built-in básicas de Python.
- Está PROHIBIDO usar open(), eval(), exec(), __import__() o acceder a \
atributos dunder (__algo__).
- Devolvé SOLO el código, sin explicación en texto, sin markdown \
(sin ```python ni ```).

Pregunta del usuario: {pregunta}
"""

_PROMPT_CORRECCION = """\
El código anterior falló al ejecutarse contra el dataset real, con este \
error:
{error}

Código que falló:
{codigo_fallido}

Corregí el código para que funcione, respetando las mismas reglas de \
antes (asignar a `resultado`, sin imports, sin open/eval/exec, solo \
código, sin markdown).
"""


def _limpiar_codigo(texto: str) -> str:
    """
    Saca los ```python / ``` si el LLM los puso a pesar de que se le
    pidió que no lo hiciera (pasa seguido, incluso con instrucciones
    explícitas en contra) -- security/sandbox.py espera código Python
    puro, no markdown.
    """
    texto = texto.strip()
    texto = re.sub(r"^```(python|py)?\s*", "", texto)
    texto = re.sub(r"\s*```$", "", texto)
    return texto.strip()


def _generar_codigo(
    pregunta: str,
    df: pd.DataFrame,
    rate_limiter: RateLimiter,
    error_previo: str | None = None,
    codigo_previo: str | None = None,
) -> str:
    """
    Le pide al LLM (vía la cadena de fallback de siempre) el código que
    responde `pregunta` sobre `df`. Si error_previo/codigo_previo vienen
    cargados, es un pedido de CORRECCIÓN tras un fallo real en la
    sandbox (no un rechazo por análisis estático -- ver analizar_con_pandasai).
    """
    prompt = _PROMPT_BASE.format(
        dtypes=df.dtypes.to_string(),
        n_filas=_FILAS_DE_MUESTRA,
        muestra=df.head(_FILAS_DE_MUESTRA).to_string(),
        pregunta=pregunta,
    )
    if error_previo and codigo_previo:
        prompt += "\n" + _PROMPT_CORRECCION.format(error=error_previo, codigo_fallido=codigo_previo)

    mensajes = [{"role": "user", "content": prompt}]

    try:
        respuesta = chat_completion(
            mensajes,
            rate_limiter=rate_limiter,
            tools=None,
            temperature=0.0,  # generación de código: priorizar determinismo
        )
    except LLMClientError as e:
        raise RuntimeError(f"No se pudo generar el código de análisis (falló toda la cadena): {e}") from e

    return _limpiar_codigo(extract_text(respuesta) or "")


def analizar_con_pandasai(
    df: pd.DataFrame,
    pregunta: str,
    rate_limiter: RateLimiter,
) -> SandboxResult:
    """
    Punto de entrada: responde una pregunta de análisis libre sobre el
    dataset. El LLM (cadena Groq -> Ollama Cloud -> Ollama local) genera
    el código; la ejecución SIEMPRE pasa por security/sandbox.py, nunca
    directo.

    Si el análisis estático de la sandbox rechaza el código (patrón
    prohibido), NO se reintenta -- ese código ya mostró intención
    sospechosa, pedirle al mismo LLM que "corrija" no tiene sentido acá
    (a diferencia de un error de ejecución normal, tipo columna
    inexistente, que sí vale la pena corregir).

    Devuelve el SandboxResult completo (success, output, error) --
    el caller (ui/) decide cómo mostrarlo.
    """
    codigo = _generar_codigo(pregunta, df, rate_limiter)
    resultado = run_sandboxed(codigo, df)

    intentos = 0
    while (
        not resultado.success
        and not resultado.rejected_by_static_analysis
        and intentos < _MAX_REINTENTOS_POR_ERROR
    ):
        intentos += 1
        codigo = _generar_codigo(pregunta, df, rate_limiter, error_previo=resultado.error, codigo_previo=codigo)
        resultado = run_sandboxed(codigo, df)

    return resultado