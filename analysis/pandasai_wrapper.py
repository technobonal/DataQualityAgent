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

NUEVO -- corregir_columnas_con_pandasai(): hasta ahora este módulo solo
servía para preguntas de SOLO LECTURA (analizar_con_pandasai devuelve
un cálculo/valor, nunca modifica el dataset). Esta función nueva es la
contraparte de ESCRITURA: cuando el pedido es una ACCIÓN de corrección
("corregí las columnas que tienen fechas guardadas como texto") en vez
de una pregunta, usa un prompt distinto que le pide al LLM modificar
`df` y devolver el DataFrame COMPLETO ya corregido en `resultado` --
porque run_sandboxed() le pasa al código una COPIA de `df` (ver
security/sandbox.py), así que mutar `df` in-place ahí adentro no alcanza
para que el cambio vuelva al caller si no se lo asigna explícitamente.

Esta función NO reemplaza a analizar_con_pandasai() ni al análisis
descriptivo que ya funciona bien (detectar y listar columnas con
tipos incorrectos) -- esa parte sigue intacta, sin tocar. Esta es una
función nueva, separada, para cuando el pedido es de corrección.

Importante -- diseño HITL: esta función devuelve el DataFrame corregido
dentro del SandboxResult, pero NO lo aplica sola a la sesión. Es
responsabilidad del caller (ui/) mostrarle al usuario qué cambió
(idealmente con un diff de dtypes antes/después) y confirmar antes de
reemplazar st.session_state.df -- misma filosofía Human-in-the-Loop que
ya usás en cleaning/, para no perder trazabilidad ni control granular.
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
{contexto}
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

# NUEVO -- prompt separado para pedidos de CORRECCIÓN (acción), distinto
# del prompt de análisis de arriba (que es de solo lectura). La
# diferencia clave: acá se exige devolver el DataFrame COMPLETO ya
# corregido en `resultado`, no un valor puntual -- ver la nota de
# "NUEVO" en el docstring del módulo sobre por qué esto es obligatorio
# (df se pasa por copia a la sandbox).
_PROMPT_BASE_CORRECCION = """\
Corregí el tipo de datos de columnas de un DataFrame pandas (`df`, ya \
cargado) según este pedido. Devolvé SOLO código, sin texto ni markdown.

Columnas y tipos: {dtypes}
Muestra ({n_filas} filas):
{muestra}
{contexto}
Pedido: {instruccion}

Reglas:
- Fechas: usá SIEMPRE corregir_fecha(df['col']) (ya disponible) en vez \
de pd.to_datetime -- este último pierde fechas válidas en formatos \
distintos al de la primera fila. Ej.: df['col'] = corregir_fecha(df['col'])
- Números: pd.to_numeric(df['col'], errors='coerce')
- No toques columnas fuera del pedido.
- Al final: resultado = df (obligatorio, DataFrame completo).
- Prohibido: imports, open/eval/exec/__import__, atributos dunder.
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


# Cuántos ítems de una lista/columnas se listan en el resumen de
# contexto (ver _resumir_resultado_previo) -- evita mandar algo enorme
# si el resultado anterior fue una lista larga.
_MAX_ITEMS_EN_RESUMEN = 20


def _resumir_resultado_previo(pregunta_previa: str, resultado_output: Any) -> str:
    """
    NUEVO -- arma un resumen CHICO (no el dataset completo) de la
    pregunta y respuesta anteriores, para que preguntas de seguimiento
    tipo "esas columnas corregilas" o "cuáles son esos valores" tengan
    con qué resolver el pronombre. Sin esto, cada pregunta es ciega a
    lo que se preguntó antes (bug real visto en producción: preguntar
    "cuáles son las columnas con outliers" y después "cuáles son los
    valores atípicos" -- la segunda no tiene forma de saber a qué
    columnas te referís).

    Se mantiene deliberadamente chico (no manda el DataFrame ni tablas
    completas) para no disparar el consumo de tokens -- ver la nota de
    tamaño de prompts de analysis/pandasai_wrapper.py.
    """
    if isinstance(resultado_output, list):
        items = resultado_output[:_MAX_ITEMS_EN_RESUMEN]
        resumen = ", ".join(str(x) for x in items)
    elif isinstance(resultado_output, pd.DataFrame):
        resumen = f"una tabla de {len(resultado_output)} filas, columnas: {list(resultado_output.columns)}"
    elif isinstance(resultado_output, pd.Series):
        resumen = f"una lista de {len(resultado_output)} valores (índice: {list(resultado_output.index)[:_MAX_ITEMS_EN_RESUMEN]})"
    else:
        resumen = str(resultado_output)[:300]

    return (
        f"\nContexto -- pregunta anterior en esta conversación (puede ser "
        f"relevante si el pedido actual usa pronombres como 'esas', "
        f"'esos', 'los mismos'):\n"
        f'  Pregunta anterior: "{pregunta_previa}"\n'
        f"  Respuesta anterior: {resumen}\n"
    )


def _generar_codigo(
    pregunta: str,
    df: pd.DataFrame,
    rate_limiter: RateLimiter,
    error_previo: str | None = None,
    codigo_previo: str | None = None,
    contexto_previo: str | None = None,
) -> str:
    """
    Le pide al LLM (vía la cadena de fallback de siempre) el código que
    responde `pregunta` sobre `df`. Si error_previo/codigo_previo vienen
    cargados, es un pedido de CORRECCIÓN tras un fallo real en la
    sandbox (no un rechazo por análisis estático -- ver analizar_con_pandasai).

    contexto_previo (opcional): resumen chico de la pregunta/respuesta
    anterior en la misma conversación de la tab -- ver
    _resumir_resultado_previo. Ya viene formateado, se inserta tal cual.
    """
    prompt = _PROMPT_BASE.format(
        dtypes=df.dtypes.to_string(),
        n_filas=_FILAS_DE_MUESTRA,
        muestra=df.head(_FILAS_DE_MUESTRA).to_string(),
        contexto=contexto_previo or "",
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


def _generar_codigo_correccion(
    instruccion: str,
    df: pd.DataFrame,
    rate_limiter: RateLimiter,
    error_previo: str | None = None,
    codigo_previo: str | None = None,
    contexto_previo: str | None = None,
) -> str:
    """
    Igual que _generar_codigo, pero usando _PROMPT_BASE_CORRECCION --
    el prompt que exige devolver el DataFrame completo corregido en
    `resultado`, en vez de un valor puntual de análisis.
    """
    prompt = _PROMPT_BASE_CORRECCION.format(
        dtypes=df.dtypes.to_string(),
        n_filas=_FILAS_DE_MUESTRA,
        muestra=df.head(_FILAS_DE_MUESTRA).to_string(),
        contexto=contexto_previo or "",
        instruccion=instruccion,
    )
    if error_previo and codigo_previo:
        prompt += "\n" + _PROMPT_CORRECCION.format(error=error_previo, codigo_fallido=codigo_previo)

    mensajes = [{"role": "user", "content": prompt}]

    try:
        respuesta = chat_completion(
            mensajes,
            rate_limiter=rate_limiter,
            tools=None,
            temperature=0.0,
        )
    except LLMClientError as e:
        raise RuntimeError(f"No se pudo generar el código de corrección (falló toda la cadena): {e}") from e

    return _limpiar_codigo(extract_text(respuesta) or "")


def analizar_con_pandasai(
    df: pd.DataFrame,
    pregunta: str,
    rate_limiter: RateLimiter,
    contexto_previo: str | None = None,
) -> SandboxResult:
    """
    Punto de entrada: responde una pregunta de análisis libre sobre el
    dataset. El LLM (cadena Groq -> Ollama Cloud -> Ollama local) genera
    el código; la ejecución SIEMPRE pasa por security/sandbox.py, nunca
    directo.

    contexto_previo (opcional): resumen de la pregunta/respuesta
    anterior en esta misma tab -- armalo con _resumir_resultado_previo
    en el caller (ui/analisis_tab.py) cuando haya una pregunta previa
    exitosa en el historial. Sin esto, cada pregunta es CIEGA a lo que
    se preguntó antes -- bug real visto en producción: preguntar
    "¿cuáles columnas tienen outliers?" y después "¿cuáles son los
    valores atípicos?" -- la segunda no tiene forma de saber a qué
    columnas te referís sin este contexto.

    Si el análisis estático de la sandbox rechaza el código (patrón
    prohibido), NO se reintenta -- ese código ya mostró intención
    sospechosa, pedirle al mismo LLM que "corrija" no tiene sentido acá
    (a diferencia de un error de ejecución normal, tipo columna
    inexistente, que sí vale la pena corregir).

    Devuelve el SandboxResult completo (success, output, error) --
    el caller (ui/) decide cómo mostrarlo.

    Esta función es de SOLO LECTURA -- nunca modifica el dataset. Para
    pedidos de corrección (acción), usar corregir_columnas_con_pandasai.
    """
    codigo = _generar_codigo(pregunta, df, rate_limiter, contexto_previo=contexto_previo)
    resultado = run_sandboxed(codigo, df)

    intentos = 0
    while (
        not resultado.success
        and not resultado.rejected_by_static_analysis
        and intentos < _MAX_REINTENTOS_POR_ERROR
    ):
        intentos += 1
        codigo = _generar_codigo(
            pregunta, df, rate_limiter,
            error_previo=resultado.error, codigo_previo=codigo,
            contexto_previo=contexto_previo,
        )
        resultado = run_sandboxed(codigo, df)

    return resultado


def corregir_columnas_con_pandasai(
    df: pd.DataFrame,
    instruccion: str,
    rate_limiter: RateLimiter,
    contexto_previo: str | None = None,
) -> SandboxResult:
    """
    NUEVO -- contraparte de escritura de analizar_con_pandasai(): para
    pedidos de ACCIÓN ("corregí las columnas que tienen fechas guardadas
    como texto"), no de preguntas de análisis.

    contexto_previo: ver la nota en analizar_con_pandasai -- mismo
    mecanismo, para pedidos tipo "esas columnas corregilas" que dependen
    de una respuesta anterior en la misma tab.

    Usa _PROMPT_BASE_CORRECCION, que le exige al LLM devolver el
    DataFrame completo ya corregido en `resultado` (no un valor
    puntual). Reintenta igual que analizar_con_pandasai si hay un error
    real de ejecución (columna inexistente, tipo de conversión
    incompatible, etc.), pero nunca si fue rechazado por el análisis
    estático.

    Validación extra respecto de analizar_con_pandasai: si el código
    "tuvo éxito" mostrar_sandboxed pero lo que devolvió NO es un
    DataFrame (el LLM se confundió y puso un texto, un número, etc.),
    se trata como fallo igual -- para eso está la corrección de tipo
    esperado, no alcanza con que la sandbox no haya tirado error.

    Devuelve el SandboxResult con el DataFrame corregido en `.output`
    si success=True. IMPORTANTE: esta función NO reemplaza el DataFrame
    de la sesión -- eso lo debe hacer el caller (ui/), idealmente
    después de mostrarle al usuario un preview del cambio (mismo
    criterio Human-in-the-Loop que cleaning/) y pedir confirmación.
    """
    codigo = _generar_codigo_correccion(instruccion, df, rate_limiter, contexto_previo=contexto_previo)
    resultado = run_sandboxed(codigo, df)

    intentos = 0
    while (
        not resultado.success
        and not resultado.rejected_by_static_analysis
        and intentos < _MAX_REINTENTOS_POR_ERROR
    ):
        intentos += 1
        codigo = _generar_codigo_correccion(
            instruccion, df, rate_limiter,
            error_previo=resultado.error, codigo_previo=codigo,
            contexto_previo=contexto_previo,
        )
        resultado = run_sandboxed(codigo, df)

    # Validación extra: si "tuvo éxito" pero no devolvió un DataFrame,
    # tratarlo como fallo -- un valor puntual (ej. un promedio) no sirve
    # acá, se esperaba el dataset completo ya corregido.
    if resultado.success and not isinstance(resultado.output, pd.DataFrame):
        resultado = SandboxResult(
            success=False,
            error=(
                "El código no devolvió un DataFrame corregido en `resultado` "
                f"(devolvió un valor de tipo {type(resultado.output).__name__} en su lugar)."
            ),
        )

    return resultado