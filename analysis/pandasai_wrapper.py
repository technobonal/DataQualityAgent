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

Seis funciones públicas, dos familias:
  - SOLO LECTURA (nunca tocan el dataset): analizar_con_pandasai(),
    generar_grafico_con_pandasai().
  - ESCRITURA (devuelven un DataFrame propuesto, HITL -- el caller
    decide si lo aplica): corregir_columnas_con_pandasai(),
    eliminar_filas_con_pandasai(), limpiar_celdas_con_pandasai(),
    eliminar_columnas_con_pandasai().

NUEVO -- eliminar_filas_con_pandasai(): misma filosofía que
corregir_columnas_con_pandasai() (prompt propio, exige devolver el
DataFrame COMPLETO en `resultado` porque la sandbox le pasa una COPIA
de `df`), pero para pedidos de eliminación de filas ("eliminá los
duplicados", "sacá las filas con nulos en la columna X"). Validación
extra específica de esta función: len(resultado) <= len(df) -- el
código generado puede reducir filas o dejarlas igual, nunca aumentarlas
(si eso pasa, es señal de que el LLM escribió algo que no es una
eliminación real, ej. un merge/concat mal armado).

NUEVO -- generar_grafico_con_pandasai(): solo lectura, como
analizar_con_pandasai(), pero en vez de pedir un valor/tabla, le pide al
LLM un gráfico usando el objeto `plt` que security/sandbox.py YA
inyecta en el entorno de ejecución (ver ese módulo -- matplotlib con
backend Agg, importado una sola vez a nivel de proceso). El código
generado nunca importa matplotlib -- ni podría, la sandbox no tiene
__import__ disponible -- solo usa la variable `plt` ya lista. El prompt
prohíbe explícitamente plt.show() (no hace nada útil sin display, y
puede colgar). Se valida que `resultado` sea una instancia real de
matplotlib.figure.Figure -- si el LLM devuelve otra cosa (ej. el `ax` en
vez del `fig`, o un string), se trata como fallo y se reintenta igual
que los otros casos.

NUEVO -- limpiar_celdas_con_pandasai(): cuarta función de escritura,
mismo patrón HITL que corregir_columnas/eliminar_filas. A diferencia de
esas dos, acá hay DOS decisiones que el LLM tiene que traducir del
pedido en lenguaje natural, no una sola:
  1. Criterio de SELECCIÓN de celdas (qué celdas tocar) -- igual de
     explícito que en _PROMPT_BASE_ELIMINACION: nulos en una columna
     puntual, o una condición concreta (ej. valores negativos).
  2. Acción a aplicar sobre esas celdas -- acotada a un enum CERRADO de
     4 opciones (rellenar_media / rellenar_mediana / rellenar_cero /
     vaciar), nunca una fórmula libre que el LLM invente -- mismo
     criterio que ya se usó para el criterio de eliminación de filas:
     "que quede registrado el criterio, no que el LLM decida
     libremente cada vez".
No se reutiliza _PROMPT_BASE_CORRECCION para esto (aunque ambas tocan
columnas existentes) porque la corrección de tipos es una operación de
CASTEO de columna completa, mientras que esto es una operación de
SELECCIÓN de celdas puntuales dentro de una columna -- son intenciones
distintas y mezclarlas en un mismo prompt genérico las volvería
ambiguas para el LLM.

NUEVO -- eliminar_columnas_con_pandasai(): quinta función de escritura,
misma filosofía que eliminar_filas_con_pandasai() pero para columnas.
Se agregó después de encontrar un bug real en producción (ver FIX más
abajo): un pedido de eliminación de COLUMNAS ("eliminá las columnas con
100% de ceros") caía en eliminar_filas_con_pandasai() porque el
clasificador de ui/analisis_tab.py solo distinguía por palabras como
"eliminá"/"sacá", y esa función NUNCA validaba que las columnas se
mantuvieran intactas -- el LLM terminó borrando columnas de verdad,
sin que ninguna validación lo frenara ni el preview lo mostrara (el
preview de esa función solo cuenta filas). Ver también el FIX en
corregir_columnas_con_pandasai() y eliminar_filas_con_pandasai() más
abajo, que cierran ese mismo agujero como red de seguridad.

FIX (esta revisión) -- validación de forma faltante en las funciones de
escritura existentes:
  - corregir_columnas_con_pandasai() solo validaba que el resultado
    fuera un DataFrame, nunca que conservara las mismas columnas Y
    filas que el original -- una corrección de TIPOS nunca debería
    agregar/sacar columnas ni filas. Ahora sí se valida.
  - eliminar_filas_con_pandasai() solo validaba que no aumentaran las
    FILAS, nunca que las COLUMNAS se mantuvieran iguales -- una
    eliminación de filas nunca debería tocar columnas. Ahora sí se
    valida.
Estas dos validaciones son la red de seguridad de fondo: aunque la
clasificación de intención en ui/analisis_tab.py se equivoque de nuevo
en el futuro (pedido ambiguo, heurística que falla), la ejecución ya no
puede aplicar silenciosamente un cambio de columnas disfrazado de
cambio de filas (o viceversa) -- se trata como error y se reintenta,
nunca se le muestra al usuario como si fuera un resultado válido.

NUEVO -- helpers de metadata (_diff_dtypes, resumen_eliminacion_filas,
resumen_relleno_celdas, resumen_eliminacion_columnas): tanto la
corrección de tipos como la eliminación de filas/columnas y el relleno
de celdas devuelven acá DataFrames "antes" y "después" -- pero
SandboxResult (definido en security/sandbox.py) es deliberadamente
genérico y no le corresponde saber de dtypes, conteo de filas/columnas,
ni conteo de celdas de este dominio. Por eso estos helpers viven ACÁ,
no en el dataclass: el caller (ui/analisis_tab.py) los llama con
(df_antes, df_después) después de que el usuario confirma el cambio, y
les da a record_action(result_summary=...) algo ya armado, sin tener
que recalcular el diff él mismo.

resumen_relleno_celdas() en particular usa DataFrame.compare() en vez
de una comparación ingenua con != -- comparar con != trata NaN != NaN
como "distinto", lo que arruinaría el conteo apenas hay una columna con
nulos que no se tocaron. compare() trata NaN == NaN como "sin cambio" y
sí detecta correctamente tanto nulo -> valor como valor -> nulo (caso
de la acción "vaciar"). Confirmado con test manual antes de escribir
esto (ver notas de la sesión).

Esta parte del módulo NO reemplaza a analizar_con_pandasai(),
corregir_columnas_con_pandasai() ni eliminar_filas_con_pandasai(), que
siguen intactas en su lógica central (solo se les agregó la validación
de forma que faltaba).

Importante -- diseño HITL: las cinco funciones de escritura
(corregir_columnas_con_pandasai, eliminar_filas_con_pandasai,
limpiar_celdas_con_pandasai, eliminar_columnas_con_pandasai) devuelven
el DataFrame propuesto dentro del SandboxResult, pero NO lo aplican
solas a la sesión. Es responsabilidad del caller (ui/) mostrarle al
usuario qué cambió y confirmar antes de reemplazar st.session_state.df
-- misma filosofía Human-in-the-Loop que ya usás en cleaning/, para no
perder trazabilidad ni control granular.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from matplotlib.figure import Figure

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

# NUEVO -- enum cerrado de acciones válidas para limpiar_celdas_con_pandasai().
# Se valida contra esta lista después de generar el código (ver
# _generar_codigo_celdas) -- si el prompt no alcanza y el LLM inventa
# una acción fuera de esta lista, se trata iguaal que cualquier otro
# error de ejecución y se reintenta.
_ACCIONES_CELDAS_VALIDAS = ("rellenar_media", "rellenar_mediana", "rellenar_cero", "vaciar")

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

# Prompt separado para pedidos de CORRECCIÓN (acción), distinto del
# prompt de análisis de arriba (que es de solo lectura). La diferencia
# clave: acá se exige devolver el DataFrame COMPLETO ya corregido en
# `resultado`, no un valor puntual -- ver la nota de "NUEVO" en el
# docstring del módulo sobre por qué esto es obligatorio (df se pasa
# por copia a la sandbox).
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
- No toques columnas fuera del pedido -- y NUNCA agregues ni elimines \
columnas o filas, esto es solo un cambio de TIPO de dato.
- Al final: resultado = df (obligatorio, DataFrame completo).
- Prohibido: imports, open/eval/exec/__import__, atributos dunder.
"""

# NUEVO -- prompt para pedidos de ELIMINACIÓN de filas. Misma lógica que
# _PROMPT_BASE_CORRECCION (exige `resultado = df` completo, no un
# subconjunto sin más contexto), pero con la regla central de esta
# función: el criterio de eliminación tiene que ser EXPLÍCITO. Un
# criterio ambiguo ("sacá las filas raras") no tiene una traducción
# única a código -- preferimos que el LLM elija la interpretación más
# literal y acotada del pedido antes que "adivine" de más.
_PROMPT_BASE_ELIMINACION = """\
Eliminá filas de un DataFrame pandas (`df`, ya cargado) según este \
pedido. Devolvé SOLO código, sin texto ni markdown.

Columnas y tipos: {dtypes}
Muestra ({n_filas} filas):
{muestra}
{contexto}
Pedido: {instruccion}

Reglas:
- Usá SIEMPRE un criterio EXPLÍCITO para eliminar filas -- nunca un \
criterio ambiguo o adivinado. Ej.: df = df.dropna(subset=['columna_x']) \
para nulos en una columna puntual, o df = df[condicion_explicita] para \
un filtro concreto, o df = df.drop_duplicates(subset=[...]) para \
duplicados.
- Si el pedido no especifica en qué columna(s) buscar nulos y dice algo \
genérico como "eliminá los nulos", aplicá dropna() sobre TODAS las \
columnas (comportamiento por defecto de pandas) -- pero nunca inventes \
una columna que no se mencionó ni que no exista en el dataset.
- El resultado NUNCA puede tener más filas que el original -- esto es \
una eliminación, no un merge ni un concat.
- PROHIBIDO tocar columnas -- esta operación SOLO saca filas. Si el \
pedido menciona "columna(s)" en vez de "fila(s)", NO es para esta \
operación -- no la ejecutes, esa es una eliminación de columnas.
- Al final: resultado = df (obligatorio, DataFrame completo tras la \
eliminación).
- Prohibido: imports, open/eval/exec/__import__, atributos dunder.
"""

# NUEVO -- prompt para pedidos de ELIMINACIÓN de columnas. Simétrico a
# _PROMPT_BASE_ELIMINACION, pero para columnas -- agregado tras el bug
# real de producción documentado en el docstring del módulo (un pedido
# de columnas ejecutándose como si fuera de filas, sin ninguna
# validación que lo frenara). El {contexto} acá es clave: normalmente
# este pedido llega como seguimiento de una pregunta anterior ("listame
# las columnas con X" -> "esas, eliminalas"), así que el bloque de
# contexto previo (ver _resumir_resultado_previo) es lo que le permite
# al LLM saber A QUÉ columnas concretas se refiere "esas".
_PROMPT_BASE_ELIMINACION_COLUMNAS = """\
Eliminá columnas de un DataFrame pandas (`df`, ya cargado) según este \
pedido. Devolvé SOLO código, sin texto ni markdown.

Columnas y tipos: {dtypes}
Muestra ({n_filas} filas):
{muestra}
{contexto}
Pedido: {instruccion}

Reglas:
- Usá SIEMPRE un criterio EXPLÍCITO para eliminar columnas -- nunca un \
criterio ambiguo o adivinado. Si el pedido se refiere a columnas \
mencionadas en el contexto de la conversación (ej. "esas", "esas \
columnas"), identificalas ahí -- NUNCA inventes un nombre de columna \
que no exista literalmente en "Columnas y tipos" de arriba.
- Ej.: df = df.drop(columns=['col_a', 'col_b']) para columnas puntuales \
por nombre, o df = df.drop(columns=df.columns[condicion_explicita]) \
para un criterio calculado (ej. más de X% de nulos).
- El resultado NUNCA puede tener más columnas que el original, ni \
columnas con nombres que no existían antes -- esto es una eliminación, \
nunca un rename ni un agregado.
- PROHIBIDO tocar filas -- esta operación SOLO saca columnas. El \
número de filas del resultado tiene que ser IDÉNTICO al original.
- Al final: resultado = df (obligatorio, DataFrame completo tras la \
eliminación).
- Prohibido: imports, open/eval/exec/__import__, atributos dunder.
"""

# NUEVO -- prompt para pedidos de GRÁFICO. Solo lectura -- nunca toca
# `df`. La diferencia central con los otros prompts: acá `resultado`
# tiene que ser un objeto Figure de matplotlib, no un valor ni un
# DataFrame, y el código usa `plt` (ya inyectado por security/sandbox.py,
# ver el docstring del módulo) en vez de importarlo.
_PROMPT_BASE_GRAFICO = """\
Generá un gráfico con matplotlib que responda a este pedido, sobre el \
DataFrame `df` ya cargado. Devolvé SOLO código, sin texto ni markdown.

Columnas y tipos: {dtypes}
Muestra ({n_filas} filas):
{muestra}
{contexto}
Pedido: {pregunta}

Reglas:
- Ya tenés disponible el objeto `plt` (matplotlib.pyplot) -- NO lo \
importes, ya está inyectado en el entorno.
- Creá la figura con fig, ax = plt.subplots() (o varios subplots si el \
pedido lo requiere).
- Al final: resultado = fig (obligatorio -- el objeto Figure completo, \
nunca `ax` ni otra cosa).
- PROHIBIDO llamar a plt.show() -- el backend no tiene display, esa \
llamada no hace nada útil y puede fallar.
- Prohibido: imports, open/eval/exec/__import__, atributos dunder.
"""

# NUEVO -- prompt para pedidos de CELDAS (rellenar o vaciar). Dos partes
# obligatorias, a diferencia de los otros prompts de escritura: el
# CRITERIO de selección (qué celdas) y la ACCIÓN sobre ellas, acotada a
# un enum cerrado -- ver _ACCIONES_CELDAS_VALIDAS y la nota "NUEVO" del
# docstring del módulo.
_PROMPT_BASE_CELDAS = """\
Modificá celdas puntuales de un DataFrame pandas (`df`, ya cargado) \
según este pedido -- rellenando valores faltantes o vaciando celdas \
existentes. Devolvé SOLO código, sin texto ni markdown.

Columnas y tipos: {dtypes}
Muestra ({n_filas} filas):
{muestra}
{contexto}
Pedido: {instruccion}

Reglas:
- Primero identificá el CRITERIO explícito de selección: en qué \
columna(s), y bajo qué condición (nulos, un valor puntual, un rango).
- Elegí UNA sola acción, de esta lista cerrada, según lo que pida el \
usuario -- no inventes una fórmula propia:
  - rellenar_media: df['col'] = df['col'].fillna(df['col'].mean())
  - rellenar_mediana: df['col'] = df['col'].fillna(df['col'].median())
  - rellenar_cero: df['col'] = df['col'].fillna(0)
  - vaciar: df.loc[condicion, 'col'] = None (para "borrar"/"vaciar" \
celdas que SÍ tienen un valor, no para rellenar)
- "rellenar_media" y "rellenar_mediana" solo tienen sentido en columnas \
numéricas -- si la columna no es numérica, usá rellenar_cero solo si \
el pedido es genérico, o "vaciar" si el pedido es borrar contenido.
- No toques filas (no elimines ninguna) ni columnas fuera del pedido.
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
    Arma un resumen CHICO (no el dataset completo) de la pregunta y
    respuesta anteriores, para que preguntas de seguimiento tipo "esas
    columnas corregilas" o "cuáles son esos valores" tengan con qué
    resolver el pronombre. Sin esto, cada pregunta es ciega a lo que se
    preguntó antes (bug real visto en producción).

    Se mantiene deliberadamente chico (no manda el DataFrame ni tablas
    completas) para no disparar el consumo de tokens.

    IMPORTANTE -- este resumen es lo que le permite al LLM resolver
    pedidos secuenciales como "listame las columnas con X" -> "esas,
    eliminalas": para una Series (caso típico de "columna: métrica",
    ej. % de nulos por columna), se listan los nombres del ÍNDICE, no
    los valores -- son los nombres los que el pedido de eliminación
    necesita para saber qué columnas tocar.
    """
    if isinstance(resultado_output, list):
        items = resultado_output[:_MAX_ITEMS_EN_RESUMEN]
        resumen = ", ".join(str(x) for x in items)
    elif isinstance(resultado_output, pd.DataFrame):
        resumen = f"una tabla de {len(resultado_output)} filas, columnas: {list(resultado_output.columns)}"
    elif isinstance(resultado_output, pd.Series):
        items = list(resultado_output.index)[:_MAX_ITEMS_EN_RESUMEN]
        resumen = f"una lista de {len(resultado_output)} valores, para estas columnas/índices: {items}"
    else:
        resumen = str(resultado_output)[:300]

    return (
        f"\nContexto -- pregunta anterior en esta conversación (puede ser "
        f"relevante si el pedido actual usa pronombres como 'esas', "
        f"'esos', 'los mismos'):\n"
        f'  Pregunta anterior: "{pregunta_previa}"\n'
        f"  Respuesta anterior: {resumen}\n"
    )


def _pedir_codigo(
    prompt_formateado: str,
    rate_limiter: RateLimiter,
    error_previo: str | None,
    codigo_previo: str | None,
) -> str:
    """
    Núcleo compartido de las cinco funciones _generar_codigo* de abajo:
    arma el mensaje final (agregando el bloque de corrección si hay un
    error previo), llama a la cadena de fallback, y limpia el código de
    la respuesta. Evita repetir este bloque idéntico cinco veces.
    """
    prompt = prompt_formateado
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
        raise RuntimeError(f"No se pudo generar el código (falló toda la cadena): {e}") from e

    return _limpiar_codigo(extract_text(respuesta) or "")


def _generar_codigo(
    pregunta: str,
    df: pd.DataFrame,
    rate_limiter: RateLimiter,
    error_previo: str | None = None,
    codigo_previo: str | None = None,
    contexto_previo: str | None = None,
) -> str:
    """Genera código de ANÁLISIS (solo lectura) -- ver _PROMPT_BASE."""
    prompt = _PROMPT_BASE.format(
        dtypes=df.dtypes.to_string(),
        n_filas=_FILAS_DE_MUESTRA,
        muestra=df.head(_FILAS_DE_MUESTRA).to_string(),
        contexto=contexto_previo or "",
        pregunta=pregunta,
    )
    return _pedir_codigo(prompt, rate_limiter, error_previo, codigo_previo)


def _generar_codigo_correccion(
    instruccion: str,
    df: pd.DataFrame,
    rate_limiter: RateLimiter,
    error_previo: str | None = None,
    codigo_previo: str | None = None,
    contexto_previo: str | None = None,
) -> str:
    """Genera código de CORRECCIÓN de tipos -- ver _PROMPT_BASE_CORRECCION."""
    prompt = _PROMPT_BASE_CORRECCION.format(
        dtypes=df.dtypes.to_string(),
        n_filas=_FILAS_DE_MUESTRA,
        muestra=df.head(_FILAS_DE_MUESTRA).to_string(),
        contexto=contexto_previo or "",
        instruccion=instruccion,
    )
    return _pedir_codigo(prompt, rate_limiter, error_previo, codigo_previo)


def _generar_codigo_eliminacion(
    instruccion: str,
    df: pd.DataFrame,
    rate_limiter: RateLimiter,
    error_previo: str | None = None,
    codigo_previo: str | None = None,
    contexto_previo: str | None = None,
) -> str:
    """NUEVO -- genera código de ELIMINACIÓN de filas -- ver _PROMPT_BASE_ELIMINACION."""
    prompt = _PROMPT_BASE_ELIMINACION.format(
        dtypes=df.dtypes.to_string(),
        n_filas=_FILAS_DE_MUESTRA,
        muestra=df.head(_FILAS_DE_MUESTRA).to_string(),
        contexto=contexto_previo or "",
        instruccion=instruccion,
    )
    return _pedir_codigo(prompt, rate_limiter, error_previo, codigo_previo)


def _generar_codigo_eliminacion_columnas(
    instruccion: str,
    df: pd.DataFrame,
    rate_limiter: RateLimiter,
    error_previo: str | None = None,
    codigo_previo: str | None = None,
    contexto_previo: str | None = None,
) -> str:
    """NUEVO -- genera código de ELIMINACIÓN de columnas -- ver _PROMPT_BASE_ELIMINACION_COLUMNAS."""
    prompt = _PROMPT_BASE_ELIMINACION_COLUMNAS.format(
        dtypes=df.dtypes.to_string(),
        n_filas=_FILAS_DE_MUESTRA,
        muestra=df.head(_FILAS_DE_MUESTRA).to_string(),
        contexto=contexto_previo or "",
        instruccion=instruccion,
    )
    return _pedir_codigo(prompt, rate_limiter, error_previo, codigo_previo)


def _generar_codigo_grafico(
    pregunta: str,
    df: pd.DataFrame,
    rate_limiter: RateLimiter,
    error_previo: str | None = None,
    codigo_previo: str | None = None,
    contexto_previo: str | None = None,
) -> str:
    """NUEVO -- genera código de GRÁFICO -- ver _PROMPT_BASE_GRAFICO."""
    prompt = _PROMPT_BASE_GRAFICO.format(
        dtypes=df.dtypes.to_string(),
        n_filas=_FILAS_DE_MUESTRA,
        muestra=df.head(_FILAS_DE_MUESTRA).to_string(),
        contexto=contexto_previo or "",
        pregunta=pregunta,
    )
    return _pedir_codigo(prompt, rate_limiter, error_previo, codigo_previo)


def _generar_codigo_celdas(
    instruccion: str,
    df: pd.DataFrame,
    rate_limiter: RateLimiter,
    error_previo: str | None = None,
    codigo_previo: str | None = None,
    contexto_previo: str | None = None,
) -> str:
    """NUEVO -- genera código de RELLENO/VACIADO de celdas -- ver _PROMPT_BASE_CELDAS."""
    prompt = _PROMPT_BASE_CELDAS.format(
        dtypes=df.dtypes.to_string(),
        n_filas=_FILAS_DE_MUESTRA,
        muestra=df.head(_FILAS_DE_MUESTRA).to_string(),
        contexto=contexto_previo or "",
        instruccion=instruccion,
    )
    return _pedir_codigo(prompt, rate_limiter, error_previo, codigo_previo)


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

    Si el análisis estático de la sandbox rechaza el código (patrón
    prohibido), NO se reintenta -- ese código ya mostró intención
    sospechosa. Un error de ejecución normal (columna inexistente, etc.)
    sí se reintenta hasta _MAX_REINTENTOS_POR_ERROR veces.

    Esta función es de SOLO LECTURA -- nunca modifica el dataset.
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
    Contraparte de escritura de analizar_con_pandasai(): para pedidos de
    ACCIÓN ("corregí las columnas que tienen fechas guardadas como
    texto"), no de preguntas de análisis.

    FIX (ver docstring del módulo): se agregó validación de que la
    FORMA se mantenga idéntica (mismas columnas, mismas filas) -- antes
    solo se validaba el tipo, lo que permitía que el LLM agregara/sacara
    columnas o filas por error sin que nada lo detectara.

    Devuelve el SandboxResult con el DataFrame corregido en `.output` si
    success=True. IMPORTANTE: esta función NO reemplaza el DataFrame de
    la sesión -- eso lo debe hacer el caller (ui/), después de mostrarle
    al usuario un preview y pedir confirmación (HITL).
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

    if resultado.success and not isinstance(resultado.output, pd.DataFrame):
        resultado = SandboxResult(
            success=False,
            error=(
                "El código no devolvió un DataFrame corregido en `resultado` "
                f"(devolvió un valor de tipo {type(resultado.output).__name__} en su lugar)."
            ),
        )
    elif resultado.success and set(resultado.output.columns) != set(df.columns):
        resultado = SandboxResult(
            success=False,
            error=(
                "El código cambió las columnas del dataset -- una corrección de "
                "tipos solo puede cambiar el DTYPE de una columna existente, "
                "nunca agregar/quitar columnas."
            ),
        )
    elif resultado.success and len(resultado.output) != len(df):
        resultado = SandboxResult(
            success=False,
            error=(
                f"El código cambió el número de filas ({len(df)} -> "
                f"{len(resultado.output)}) -- una corrección de tipos nunca "
                "debería agregar ni quitar filas."
            ),
        )

    return resultado


def eliminar_filas_con_pandasai(
    df: pd.DataFrame,
    instruccion: str,
    rate_limiter: RateLimiter,
    contexto_previo: str | None = None,
) -> SandboxResult:
    """
    NUEVO -- tercera función de escritura del módulo (junto a
    corregir_columnas_con_pandasai): para pedidos de ELIMINACIÓN de
    filas ("sacá los duplicados", "eliminá las filas con nulos en
    edad"). Mismo patrón HITL: devuelve el DataFrame propuesto, nunca lo
    aplica sola.

    Usa _PROMPT_BASE_ELIMINACION, que exige un criterio explícito (nunca
    ambiguo) y devolver el DataFrame completo tras la eliminación.

    Validaciones:
      1. El resultado tiene que ser un DataFrame (igual que corrección).
      2. len(resultado) <= len(df) -- una eliminación real nunca puede
         terminar con MÁS filas que las que había. Si el LLM generó
         código que de alguna forma agregó filas (ej. un merge/concat
         mal armado en vez de un filtro), se trata como fallo y se
         reintenta -- no se le pasa esto al usuario como si fuera un
         resultado válido.
      3. FIX (ver docstring del módulo): las COLUMNAS tienen que ser
         idénticas a las del original -- esta es la validación que
         faltaba y permitió el bug real de producción donde un pedido
         de eliminación de columnas se ejecutó acá y borró columnas sin
         que nada lo frenara ni el preview lo mostrara (el preview de
         esta función solo cuenta filas).

    Para armar la metadata de record_action() (filas_antes,
    filas_despues, criterio_aplicado, muestra de filas eliminadas), usar
    resumen_eliminacion_filas(df, resultado.output, instruccion) después
    de que el usuario confirme -- ver ese helper más abajo.
    """
    codigo = _generar_codigo_eliminacion(instruccion, df, rate_limiter, contexto_previo=contexto_previo)
    resultado = run_sandboxed(codigo, df)

    intentos = 0
    while (
        not resultado.success
        and not resultado.rejected_by_static_analysis
        and intentos < _MAX_REINTENTOS_POR_ERROR
    ):
        intentos += 1
        codigo = _generar_codigo_eliminacion(
            instruccion, df, rate_limiter,
            error_previo=resultado.error, codigo_previo=codigo,
            contexto_previo=contexto_previo,
        )
        resultado = run_sandboxed(codigo, df)

    if resultado.success and not isinstance(resultado.output, pd.DataFrame):
        resultado = SandboxResult(
            success=False,
            error=(
                "El código no devolvió un DataFrame en `resultado` "
                f"(devolvió un valor de tipo {type(resultado.output).__name__} en su lugar)."
            ),
        )
    elif resultado.success and len(resultado.output) > len(df):
        resultado = SandboxResult(
            success=False,
            error=(
                f"El código aumentó el número de filas ({len(df)} -> "
                f"{len(resultado.output)}) en vez de eliminarlas -- esto no es "
                "una eliminación válida."
            ),
        )
    elif resultado.success and set(resultado.output.columns) != set(df.columns):
        resultado = SandboxResult(
            success=False,
            error=(
                "El código cambió las columnas del dataset -- una eliminación de "
                "FILAS nunca debería tocar columnas. Si el pedido era sobre "
                "columnas, no es esta operación."
            ),
        )

    return resultado


def eliminar_columnas_con_pandasai(
    df: pd.DataFrame,
    instruccion: str,
    rate_limiter: RateLimiter,
    contexto_previo: str | None = None,
) -> SandboxResult:
    """
    NUEVO -- quinta función de escritura del módulo, simétrica a
    eliminar_filas_con_pandasai() pero para columnas ("eliminá las
    columnas con más del 60% de nulos", o de seguimiento: "esas,
    eliminalas" después de haber preguntado por columnas). Mismo patrón
    HITL: devuelve el DataFrame propuesto, nunca lo aplica sola.

    Usa _PROMPT_BASE_ELIMINACION_COLUMNAS, que exige un criterio
    explícito y aprovecha `contexto_previo` para resolver pedidos de
    seguimiento que se refieren a columnas ya mencionadas antes en la
    conversación.

    Validaciones (simétricas a eliminar_filas_con_pandasai):
      1. El resultado tiene que ser un DataFrame.
      2. Ninguna columna nueva -- set(resultado.columns) tiene que ser
         subconjunto de set(df.columns). Si aparece un nombre de
         columna que no existía, es señal de que el LLM escribió un
         rename o alguna transformación que no es una eliminación real.
      3. len(resultado.columns) <= len(df.columns) -- nunca puede haber
         más columnas que antes.
      4. Las FILAS tienen que ser idénticas en cantidad al original --
         esta operación solo puede sacar columnas, nunca filas.

    Para armar la metadata de record_action() (columnas_antes,
    columnas_despues, columnas_eliminadas), usar
    resumen_eliminacion_columnas(df, resultado.output, instruccion)
    después de que el usuario confirme -- ver ese helper más abajo.
    """
    codigo = _generar_codigo_eliminacion_columnas(instruccion, df, rate_limiter, contexto_previo=contexto_previo)
    resultado = run_sandboxed(codigo, df)

    intentos = 0
    while (
        not resultado.success
        and not resultado.rejected_by_static_analysis
        and intentos < _MAX_REINTENTOS_POR_ERROR
    ):
        intentos += 1
        codigo = _generar_codigo_eliminacion_columnas(
            instruccion, df, rate_limiter,
            error_previo=resultado.error, codigo_previo=codigo,
            contexto_previo=contexto_previo,
        )
        resultado = run_sandboxed(codigo, df)

    if resultado.success and not isinstance(resultado.output, pd.DataFrame):
        resultado = SandboxResult(
            success=False,
            error=(
                "El código no devolvió un DataFrame en `resultado` "
                f"(devolvió un valor de tipo {type(resultado.output).__name__} en su lugar)."
            ),
        )
    elif resultado.success and not set(resultado.output.columns).issubset(set(df.columns)):
        columnas_inventadas = set(resultado.output.columns) - set(df.columns)
        resultado = SandboxResult(
            success=False,
            error=(
                f"El código devolvió columnas que no existían en el original "
                f"({sorted(columnas_inventadas)}) -- esto no es una eliminación "
                "válida (parece un rename o una transformación distinta)."
            ),
        )
    elif resultado.success and len(resultado.output.columns) > len(df.columns):
        resultado = SandboxResult(
            success=False,
            error=(
                f"El código aumentó el número de columnas ({len(df.columns)} -> "
                f"{len(resultado.output.columns)}) en vez de eliminarlas."
            ),
        )
    elif resultado.success and len(resultado.output) != len(df):
        resultado = SandboxResult(
            success=False,
            error=(
                f"El código cambió el número de filas ({len(df)} -> "
                f"{len(resultado.output)}) -- una eliminación de COLUMNAS nunca "
                "debería tocar filas."
            ),
        )

    return resultado


def generar_grafico_con_pandasai(
    df: pd.DataFrame,
    pregunta: str,
    rate_limiter: RateLimiter,
    contexto_previo: str | None = None,
) -> SandboxResult:
    """
    NUEVO -- solo lectura, como analizar_con_pandasai(): genera un
    gráfico matplotlib a partir de un pedido en lenguaje natural. Usa
    _PROMPT_BASE_GRAFICO, que aprovecha el objeto `plt` que
    security/sandbox.py ya inyecta (backend Agg, sin import necesario).

    Validación: `resultado` tiene que ser una instancia real de
    matplotlib.figure.Figure -- si el LLM devuelve el `ax` en vez del
    `fig`, o cualquier otra cosa, se trata como fallo y se reintenta
    igual que un error de ejecución común.

    Nunca modifica el dataset -- el caller (ui/) renderiza directo con
    st.pyplot(resultado.output), sin pasar por record_action() ni por
    ningún flujo de confirmación (no hay nada que confirmar, no se toca
    el dataset).
    """
    codigo = _generar_codigo_grafico(pregunta, df, rate_limiter, contexto_previo=contexto_previo)
    resultado = run_sandboxed(codigo, df)

    intentos = 0
    while (
        not resultado.success
        and not resultado.rejected_by_static_analysis
        and intentos < _MAX_REINTENTOS_POR_ERROR
    ):
        intentos += 1
        codigo = _generar_codigo_grafico(
            pregunta, df, rate_limiter,
            error_previo=resultado.error, codigo_previo=codigo,
            contexto_previo=contexto_previo,
        )
        resultado = run_sandboxed(codigo, df)

    if resultado.success and not isinstance(resultado.output, Figure):
        resultado = SandboxResult(
            success=False,
            error=(
                "El código no devolvió una Figure de matplotlib en `resultado` "
                f"(devolvió un valor de tipo {type(resultado.output).__name__} en su lugar). "
                "Recordá: resultado = fig, no ax."
            ),
        )

    return resultado


def limpiar_celdas_con_pandasai(
    df: pd.DataFrame,
    instruccion: str,
    rate_limiter: RateLimiter,
    contexto_previo: str | None = None,
) -> SandboxResult:
    """
    NUEVO -- cuarta función de escritura del módulo: para pedidos de
    RELLENO o VACIADO de celdas puntuales ("rellená los nulos de
    ventas con la media", "vaciá el precio donde sea negativo"). Mismo
    patrón HITL que las otras: devuelve el DataFrame propuesto, nunca lo
    aplica sola.

    Usa _PROMPT_BASE_CELDAS, que exige un criterio de selección
    explícito Y una acción de un enum cerrado (ver
    _ACCIONES_CELDAS_VALIDAS) -- a diferencia de corregir_columnas y
    eliminar_filas, acá no se valida el enum contra el CÓDIGO generado
    (sería frágil, el LLM puede escribir la misma operación de varias
    formas válidas de Python) sino contra el RESULTADO: si ninguna
    celda cambió, se trata como fallo -- señal de que el criterio de
    selección no encontró nada, o el LLM no aplicó ninguna acción real.

    Validaciones:
      1. El resultado tiene que ser un DataFrame (igual que las otras).
      2. La FORMA tiene que ser idéntica a la del original (mismas filas
         y columnas) -- esta operación solo puede tocar valores de
         celdas existentes, nunca agregar/sacar filas o columnas. Si
         cambia la forma, es señal de que el LLM hizo algo que no es
         esta operación (ej. un dropna() en vez de un fillna()).

    Para armar la metadata de record_action() (columnas afectadas,
    cantidad de celdas tocadas), usar
    resumen_relleno_celdas(df, resultado.output, instruccion) después de
    que el usuario confirme -- ver ese helper más abajo.
    """
    codigo = _generar_codigo_celdas(instruccion, df, rate_limiter, contexto_previo=contexto_previo)
    resultado = run_sandboxed(codigo, df)

    intentos = 0
    while (
        not resultado.success
        and not resultado.rejected_by_static_analysis
        and intentos < _MAX_REINTENTOS_POR_ERROR
    ):
        intentos += 1
        codigo = _generar_codigo_celdas(
            instruccion, df, rate_limiter,
            error_previo=resultado.error, codigo_previo=codigo,
            contexto_previo=contexto_previo,
        )
        resultado = run_sandboxed(codigo, df)

    if resultado.success and not isinstance(resultado.output, pd.DataFrame):
        resultado = SandboxResult(
            success=False,
            error=(
                "El código no devolvió un DataFrame en `resultado` "
                f"(devolvió un valor de tipo {type(resultado.output).__name__} en su lugar)."
            ),
        )
    elif resultado.success and resultado.output.shape != df.shape:
        resultado = SandboxResult(
            success=False,
            error=(
                f"El código cambió la forma del dataset ({df.shape} -> "
                f"{resultado.output.shape}) -- esta operación solo puede modificar "
                "valores de celdas existentes, no agregar ni quitar filas/columnas."
            ),
        )

    return resultado


def _diff_dtypes(df_antes: pd.DataFrame, df_despues: pd.DataFrame) -> dict[str, dict[str, str]]:
    """
    Helper de metadata para corregir_columnas_con_pandasai(): compara
    los dtypes de `df_antes` y `df_despues` y devuelve SOLO las columnas
    que efectivamente cambiaron de tipo -- listo para pasarle a
    record_action(result_summary=...) sin que la UI tenga que calcularlo.

    Formato: {"columna": {"antes": "object", "despues": "datetime64[ns]"}, ...}
    """
    cambios: dict[str, dict[str, str]] = {}
    for columna in df_antes.columns:
        if columna not in df_despues.columns:
            continue
        antes = str(df_antes[columna].dtype)
        despues = str(df_despues[columna].dtype)
        if antes != despues:
            cambios[columna] = {"antes": antes, "despues": despues}
    return cambios


def resumen_eliminacion_filas(
    df_antes: pd.DataFrame,
    df_despues: pd.DataFrame,
    instruccion: str,
) -> dict[str, Any]:
    """
    NUEVO -- helper de metadata análogo a _diff_dtypes(), pero para
    eliminar_filas_con_pandasai(): arma todo lo que necesita
    record_action(result_summary=...) y el preview de confirmación de la
    UI, sin que ui/analisis_tab.py tenga que recalcular el diff de
    índices él mismo.

    `criterio_aplicado` es la instrucción original del usuario (en sus
    propias palabras) -- no el código Python generado, que es un detalle
    de implementación y no siempre es legible para el usuario final en
    el preview.

    `df_eliminadas` son las filas que efectivamente se sacaron (vía
    diferencia de índices) -- la UI puede hacer .head(N) sobre esto con
    el N que prefiera para la muestra del preview, en vez de recalcular
    la resta acá.
    """
    indices_eliminados = df_antes.index.difference(df_despues.index)
    return {
        "filas_antes": len(df_antes),
        "filas_despues": len(df_despues),
        "filas_eliminadas": len(indices_eliminados),
        "criterio_aplicado": instruccion,
        "df_eliminadas": df_antes.loc[indices_eliminados],
    }


def resumen_eliminacion_columnas(
    df_antes: pd.DataFrame,
    df_despues: pd.DataFrame,
    instruccion: str,
) -> dict[str, Any]:
    """
    NUEVO -- helper de metadata análogo a resumen_eliminacion_filas(),
    para eliminar_columnas_con_pandasai(): arma todo lo que necesita
    record_action(result_summary=...) y el preview de confirmación de la
    UI, sin que ui/analisis_tab.py tenga que recalcular la diferencia de
    columnas él mismo.

    `columnas_eliminadas` se calcula preservando el ORDEN original de
    `df_antes` (no un set desordenado) -- más prolijo para mostrar en el
    preview y para el log de auditoría.
    """
    columnas_antes = list(df_antes.columns)
    columnas_despues = list(df_despues.columns)
    columnas_eliminadas = [c for c in columnas_antes if c not in columnas_despues]
    return {
        "columnas_antes": columnas_antes,
        "columnas_despues": columnas_despues,
        "columnas_eliminadas": columnas_eliminadas,
        "criterio_aplicado": instruccion,
    }


def resumen_relleno_celdas(
    df_antes: pd.DataFrame,
    df_despues: pd.DataFrame,
    instruccion: str,
) -> dict[str, Any]:
    """
    NUEVO -- helper de metadata análogo a resumen_eliminacion_filas(),
    para limpiar_celdas_con_pandasai(): cuenta cuántas celdas cambiaron
    por columna, y arma lo que necesita record_action(result_summary=...)
    y el preview de confirmación de la UI.

    Usa DataFrame.compare() en vez de una comparación con != -- ver la
    nota "NUEVO" en el docstring del módulo sobre por qué != rompe el
    conteo con columnas que tienen NaN.

    `columnas_afectadas` es un dict {"columna": cantidad_de_celdas} --
    solo incluye columnas donde efectivamente cambió algo (compare()
    solo devuelve columnas con al menos una diferencia).
    """
    cambios = df_antes.compare(df_despues)

    columnas_afectadas: dict[str, int] = {}
    if not cambios.empty:
        for columna in cambios.columns.get_level_values(0).unique():
            # dropna(how="all") -- una fila puede aparecer en `cambios`
            # por una diferencia en OTRA columna del mismo df.compare();
            # esto cuenta solo las filas donde ESTA columna puntual
            # efectivamente cambió (en cualquier dirección: valor->NaN
            # incluido, no solo NaN->valor).
            columnas_afectadas[str(columna)] = int(len(cambios[columna].dropna(how="all")))

    return {
        "columnas_afectadas": columnas_afectadas,
        "celdas_afectadas_total": sum(columnas_afectadas.values()),
        "criterio_aplicado": instruccion,
    }

def renombrar_columna(df: pd.DataFrame, columna_actual: str, nuevo_nombre: str) -> pd.DataFrame:
            """
            NUEVO -- sexta función de escritura del módulo, pero a diferencia de
            las cinco anteriores esta es DETERMINÍSTICA -- no pasa por el LLM ni
            por run_sandboxed(). Renombrar una columna es una operación sin
            ambigüedad (df.rename()), meterla por el LLM sería latencia y costo
            de tokens sin ningún beneficio, con el riesgo extra de que el modelo
            alucine el nombre.

            Devuelve una COPIA del DataFrame con la columna renombrada -- `df`
            original queda intacto (mismo contrato que las demás funciones de
            escritura: el caller decide cuándo reemplazar sesion.current_df,
            vía record_action(), después de que el usuario confirme en el
            preview).

            Lanza ValueError (no SandboxResult -- acá no hay ejecución de código
            de terceros que rechazar/reintentar, solo validación de argumentos)
            si:
            - `columna_actual` no existe en el dataset.
            - `nuevo_nombre` queda vacío después de un .strip().
            - `nuevo_nombre` ya es el nombre de otra columna existente (evita
                una colisión silenciosa que pisaría datos).
            El caller (ui/) captura esto con un try/except y lo muestra con
            st.error(), mismo patrón que ya usás alrededor de las funciones
            _con_pandasai en ui/analisis_tab.py.
            """
            nuevo_nombre = nuevo_nombre.strip()
            if columna_actual not in df.columns:
                raise ValueError(f"La columna '{columna_actual}' no existe en el dataset.")
            if not nuevo_nombre:
                raise ValueError("El nuevo nombre de columna no puede estar vacío.")
            if nuevo_nombre != columna_actual and nuevo_nombre in df.columns:
                raise ValueError(f"Ya existe una columna llamada '{nuevo_nombre}'.")
            return df.rename(columns={columna_actual: nuevo_nombre})


def agregar_columna_vacia(df: pd.DataFrame, nombre_columna: str) -> pd.DataFrame:
    """
    NUEVO -- séptima función de escritura, también determinística.
    Agrega una columna nueva al final del dataset, con todos los
    valores en NA (pd.NA) -- para completarse después con otra
    herramienta (limpiar_celdas_con_pandasai, corregir_columnas, o
    edición manual), no con un valor ya decidido acá.

    Devuelve una COPIA del DataFrame -- `df` original queda intacto.

    Lanza ValueError si:
      - `nombre_columna` queda vacío después de un .strip().
      - `nombre_columna` ya existe en el dataset (evita pisar una
        columna real sin querer).
    """
    nombre_columna = nombre_columna.strip()
    if not nombre_columna:
        raise ValueError("El nombre de la columna nueva no puede estar vacío.")
    if nombre_columna in df.columns:
        raise ValueError(f"Ya existe una columna llamada '{nombre_columna}'.")
    df_nuevo = df.copy()
    df_nuevo[nombre_columna] = pd.NA
    return df_nuevo

def agregar_columna_vacia(
    df: pd.DataFrame,
    nombre_columna: str,
    entre_columnas: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """
    NUEVO -- séptima función de escritura, determinística. Agrega una
    columna nueva al dataset, con todos los valores en NA (pd.NA) --
    para completarse después con otra herramienta (limpiar_celdas,
    corregir_columnas, o edición manual), no con un valor ya decidido
    acá. Devuelve una COPIA del DataFrame -- `df` original intacto.

    Posición de inserción:
      - entre_columnas=None (default): se agrega al final, mismo
        comportamiento simple de antes.
      - entre_columnas=(columna_izquierda, columna_derecha): se inserta
        EXACTAMENTE en esa posición, usando df.insert() en vez de
        reconstruir el orden de columnas a mano. Se exige que las dos
        columnas dadas sean ADYACENTES en el orden ACTUAL del dataset
        (columna_derecha tiene que ser la próxima columna después de
        columna_izquierda) -- si no lo son, "entre" queda ambiguo
        (¿antes de qué columna intermedia?), así que se rechaza con
        ValueError en vez de adivinar una posición. Esto es lo que la
        UI le va a preguntar al usuario: entre qué dos columnas
        adyacentes va la nueva.

    Lanza ValueError si: el nombre queda vacío, ya existe, alguna de
    las dos columnas de referencia no existe, o no son adyacentes.
    """
    nombre_columna = nombre_columna.strip()
    if not nombre_columna:
        raise ValueError("El nombre de la columna nueva no puede estar vacío.")
    if nombre_columna in df.columns:
        raise ValueError(f"Ya existe una columna llamada '{nombre_columna}'.")

    df_nuevo = df.copy()

    if entre_columnas is None:
        df_nuevo[nombre_columna] = pd.NA
        return df_nuevo

    columna_izquierda, columna_derecha = entre_columnas
    columnas = list(df.columns)

    if columna_izquierda not in columnas:
        raise ValueError(f"La columna '{columna_izquierda}' no existe en el dataset.")
    if columna_derecha not in columnas:
        raise ValueError(f"La columna '{columna_derecha}' no existe en el dataset.")

    idx_izquierda = columnas.index(columna_izquierda)
    idx_derecha = columnas.index(columna_derecha)
    if idx_derecha != idx_izquierda + 1:
        raise ValueError(
            f"'{columna_izquierda}' y '{columna_derecha}' no son columnas "
            "adyacentes en el dataset -- para insertar 'entre' dos columnas, "
            "tienen que estar una al lado de la otra en el orden actual."
        )

    df_nuevo.insert(loc=idx_izquierda + 1, column=nombre_columna, value=pd.NA)
    return df_nuevo


def resumen_columna_agregada(
    nombre_columna: str,
    entre_columnas: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """NUEVO -- helper de metadata para agregar_columna_vacia()."""
    resumen: dict[str, Any] = {"columna_agregada": nombre_columna}
    resumen["posicion"] = (
        f"entre '{entre_columnas[0]}' y '{entre_columnas[1]}'"
        if entre_columnas is not None
        else "al final del dataset"
    )
    return resumen


def resumen_renombrado_columna(columna_actual: str, nuevo_nombre: str) -> dict[str, Any]:
    """
    NUEVO -- helper de metadata para renombrar_columna(), análogo a
    _diff_dtypes() y los demás resumen_*: arma lo que necesita
    record_action(result_summary=...) y el preview de confirmación de
    la UI. No hace falta comparar df_antes/df_después acá -- el cambio
    ya es explícito en los dos argumentos.
    """
    return {"columna_actual": columna_actual, "nuevo_nombre": nuevo_nombre}


def resumen_columna_agregada(
    nombre_columna: str,
    entre_columnas: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """NUEVO -- helper de metadata para agregar_columna_vacia()."""
    resumen: dict[str, Any] = {"columna_agregada": nombre_columna}
    resumen["posicion"] = (
        f"entre '{entre_columnas[0]}' y '{entre_columnas[1]}'"
        if entre_columnas is not None
        else "al final del dataset"
    )
    return resumen