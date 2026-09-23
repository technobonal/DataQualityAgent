"""
security/sandbox.py

Sandbox de ejecución para código Python generado dinámicamente — pensado
para analysis/pandasai_wrapper.py, que le pide a un LLM que genere código
pandas a partir de una pregunta en lenguaje natural, y lo ejecuta.

Ejecutar código generado por un LLM es un riesgo real: el modelo podría
(por error, o por un prompt malicioso metido en el dataset) generar
código que borre archivos, haga llamadas de red, o lea variables de
entorno con API keys. Este módulo aplica dos capas de defensa:

  1. Análisis estático: rechaza el código ANTES de ejecutarlo si contiene
     patrones prohibidos (imports peligrosos, acceso a archivos, eval/exec, etc.).
  2. Ejecución restringida: corre el código en un PROCESO APARTE, con un
     conjunto de builtins reducido (sin open, __import__, eval, exec) y
     con un límite de tiempo — si se cuelga o tarda de más, se mata el
     proceso sin afectar el resto de la aplicación.

Aclaración honesta: esto NO es una sandbox a prueba de balas — ningún
sandbox puro en Python lo es al 100% (un atacante suficientemente
creativo siempre puede buscar vueltas). Es defensa en profundidad
razonable para este proyecto: el código ejecutado viene de un LLM que
solo puede ver pandas y el propio DataFrame, no un entorno multi-usuario
expuesto a internet sin control.

FIX (ver analysis/pandasai_wrapper.py, bug del "None" al pedir
correcciones de columnas): antes, si el código corría sin tirar
excepción pero SE OLVIDABA de asignar `resultado` (o `result`), esto se
reportaba como ÉXITO con output=None -- indistinguible de un cálculo
legítimo que dé None. Ahora se chequea explícitamente si la variable fue
asignada; si no, se devuelve como FALLO (con un mensaje claro), para que
el caller (pandasai_wrapper.analizar_con_pandasai /
corregir_columnas_con_pandasai) pueda reintentar pidiéndole al LLM que
corrija el código, en vez de creer que el análisis realmente dio None.

Soporte de gráficos (matplotlib): el código generado no puede hacer
`import matplotlib` -- ni falta hace agregarlo a _PATRONES_PROHIBIDOS,
porque el entorno restringido ya no tiene `__import__` disponible (ver
_BUILTINS_PERMITIDOS), así que cualquier `import` de lo que sea falla en
tiempo de ejecución de todos modos. En cambio, mismo patrón ya usado con
`corregir_fecha`: se importa matplotlib ACÁ (a nivel de módulo, una sola
vez, con backend "Agg" -- no interactivo, obligatorio en un proceso sin
display; si se usa el backend por defecto puede colgarse o fallar) y se
inyecta el objeto `plt` ya listo en el entorno de ejecución. El código
generado simplemente lo usa como variable disponible: `fig, ax =
plt.subplots()`, nunca lo importa ni lo instancia él mismo.

Nota de picklabilidad: un objeto Figure de matplotlib viaja de vuelta al
proceso padre por la misma multiprocessing.Queue que ya se usa para
DataFrames -- con backend Agg (sin canvas interactivo/GUI) las Figure
son picklables en la práctica; confirmado con test manual (ver
tests/test_sandbox.py, pendiente de escribir formalmente).

FIX (revisión de seguridad post-implementación, 3 hallazgos):

  1. _BUILTINS_PERMITIDOS no incluía ningún tipo de excepción
     (Exception, ValueError, KeyError, etc.) ni isinstance/type.
     Código generado perfectamente razonable -- un try/except, un
     isinstance() -- reventaba con NameError en vez de con el error
     real. Confirmado con testing manual antes de este fix.

  2. _PATRONES_PROHIBIDOS solo cubría `import X` literal, no
     `from X import Y`. Ejemplo: "from os import listdir" pasaba el
     análisis estático sin ser rechazado (rejected_by_static_analysis
     quedaba en False). No era explotable -- la segunda capa (sin
     __import__ en builtins) lo frenaba igual en runtime -- pero es
     el tipo de gap que un día deja de estar cubierto si alguien toca
     la segunda capa sin saber que el análisis estático dependía de
     ella. Se agregó el patrón correspondiente.

  3. IMPORTANTE -- acoplamiento implícito entre `type` y el regex de
     dunders: agregar `type` a los builtins reabre, en teoría, el
     escape clásico de sandboxes en Python:
         objeto.__class__.__bases__[0].__subclasses__()
     que permite llegar a clases ya cargadas en memoria (algunas
     potencialmente peligrosas) SIN necesidad de import. Esto sigue
     bloqueado porque el regex `__[a-zA-Z_]+__` prohíbe cualquier
     acceso a atributos dunder como texto -- pero es una dependencia
     que antes no estaba documentada. NO QUITAR ese patrón del regex
     sin agregar antes una defensa equivalente (ej. un AST-walker que
     rechace Attribute nodes dunder) -- si se quita, este escape
     vuelve a estar abierto.
"""

from __future__ import annotations

import builtins as _builtins_module
import multiprocessing
import re
from dataclasses import dataclass
from typing import Any

import matplotlib
matplotlib.use("Agg")  # backend no interactivo -- obligatorio en un proceso sin display
import matplotlib.pyplot as plt

import pandas as pd

# Tiempo máximo (segundos) que se le permite correr al código generado
# antes de matar el proceso — evita loops infinitos o cómputos colgados.
_TIMEOUT_SEGUNDOS = 10

# Patrones de texto prohibidos en el código generado — primera línea de
# defensa (análisis estático), antes de intentar ejecutar nada.
_PATRONES_PROHIBIDOS = [
    r"\bimport\s+os\b", r"\bimport\s+sys\b", r"\bimport\s+subprocess\b",
    r"\bimport\s+socket\b", r"\bimport\s+shutil\b", r"\bimport\s+requests\b",
    r"\bimport\s+urllib\b", r"\bimport\s+pathlib\b", r"\bimport\s+ctypes\b",
    # FIX #2 -- "from X import Y" no matcheaba ningún patrón de arriba.
    # No explotable hoy (sin __import__ en builtins, falla igual en
    # runtime) pero se cierra acá también para no depender solo de esa
    # segunda capa.
    r"\bfrom\s+\w+\s+import\b",
    r"__import__\s*\(", r"\bopen\s*\(", r"\beval\s*\(", r"\bexec\s*\(",
    r"\bcompile\s*\(", r"\bgetattr\s*\(", r"\bsetattr\s*\(", r"\bdelattr\s*\(",
    r"\bglobals\s*\(", r"\blocals\s*\(", r"\bvars\s*\(",
    # Cualquier acceso a dunder attributes/métodos -- ESTE patrón es lo
    # único que neutraliza el escape vía type()/__class__ ahora que
    # type está en _BUILTINS_PERMITIDOS (ver FIX #3 arriba). No tocar
    # sin agregar una defensa equivalente.
    r"__[a-zA-Z_]+__",
    r"\bos\.", r"\bsys\.", r"\bsubprocess\.",
]

# Builtins mínimos necesarios para operaciones normales de pandas/cálculo.
# Todo lo que no está en esta lista (open, __import__, eval, exec, input,
# etc.) directamente no existe dentro del código ejecutado.
_BUILTINS_PERMITIDOS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "int", "isinstance", "len", "list", "map", "max", "min", "print",
    "range", "reversed", "round", "set", "sorted", "str", "sum", "tuple",
    "type", "zip",
    "True", "False", "None",
    # FIX #1 -- excepciones básicas. Sin esto, cualquier try/except en
    # código generado por el LLM (patrón normal y esperable) reventaba
    # con NameError en vez de con el error real que se quiso atrapar.
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "AttributeError", "ZeroDivisionError", "StopIteration",
}


def _corregir_columna_fecha_robusta(serie: pd.Series) -> pd.Series:
    """
    Convierte una columna de texto a datetime.

    HISTORIA DEL DISEÑO (dos iteraciones):

    v1 (parseo valor por valor con pd.to_datetime): resolvía bien el
    problema de pérdida de datos (pd.to_datetime en LOTE adivina el
    formato mirando el primer valor no nulo y lo fuerza sobre el resto,
    perdiendo fechas válidas en otro formato) -- pero era demasiado
    lento en datasets reales: ~4.3s para 20.000 filas, lo cual en la
    práctica superaba el timeout de 10s de la sandbox cuando el LLM
    intentaba corregir más de una columna de fecha en el mismo código
    (bug real visto en producción: "El código tardó más de 10s y fue
    detenido").

    v2 (versión ACTUAL): prueba una lista de formatos EXPLÍCITOS
    comunes de forma VECTORIZADA (cada intento es prácticamente gratis,
    ~ms, porque se aplica a la columna entera de una vez en vez de
    valor por valor) -- y solo lo que queda sin resolver después de
    probar todos los formatos (normalmente muy poco: texto que
    realmente no es fecha, o un formato raro no listado) cae al modo
    lento fila-por-fila, pero sobre ese subconjunto chico, no sobre
    toda la columna. Resultado: mismo nivel de recuperación de datos
    que v1, pero ~300x más rápido en el benchmark de 20.000 filas
    (0.014s vs 4.33s).
    """
    resultado = pd.Series(pd.NaT, index=serie.index, dtype="datetime64[ns]")
    pendientes = serie.notna()

    _FORMATOS_COMUNES = (
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
        "%d-%m-%Y", "%Y.%m.%d", "%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S",
    )
    for fmt in _FORMATOS_COMUNES:
        if not pendientes.any():
            break
        intento = pd.to_datetime(serie[pendientes], format=fmt, errors="coerce")
        resueltos = intento.notna()
        resultado.loc[intento.index[resueltos]] = intento[resueltos]
        pendientes.loc[intento.index[resueltos]] = False

    if pendientes.any():
        recuperados = serie[pendientes].apply(lambda v: pd.to_datetime(v, errors="coerce"))
        resultado.loc[pendientes] = recuperados

    return resultado

# Centinela interno para distinguir "la variable resultado/result existe
# y su valor es None" (caso legítimo) de "la variable nunca se asignó"
# (el bug que causaba el None fantasma). No es visible fuera de este
# módulo -- nunca se expone en SandboxResult.output.
_NO_ASIGNADO = object()


@dataclass
class SandboxResult:
    """Resultado de una ejecución en la sandbox."""
    success: bool
    output: Any = None
    error: str | None = None
    rejected_by_static_analysis: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serializa a dict plano — convierte el output a string si no es JSON-serializable."""
        salida = self.output
        if not isinstance(salida, (str, int, float, bool, type(None), list, dict)):
            salida = str(salida)
        return {
            "success": self.success,
            "output": salida,
            "error": self.error,
            "rejected_by_static_analysis": self.rejected_by_static_analysis,
        }


def analizar_codigo_estaticamente(codigo: str) -> str | None:
    """
    Revisa el código en busca de patrones prohibidos ANTES de ejecutarlo.
    Devuelve el patrón encontrado (string) si el código es rechazado, o
    None si pasó la revisión.
    """
    for patron in _PATRONES_PROHIBIDOS:
        if re.search(patron, codigo):
            return patron
    return None


def _ejecutar_en_proceso_aislado(codigo: str, df: pd.DataFrame, queue: multiprocessing.Queue) -> None:
    """
    Corre DENTRO del proceso hijo aislado. Arma un entorno con builtins
    reducidos y solo pandas/el DataFrame disponibles, y ejecuta el
    código ahí. El resultado (o el error) se manda de vuelta al proceso
    padre por la queue — nunca se comparte memoria directamente.
    """
    builtins_restringidos = {
        nombre: getattr(_builtins_module, nombre)
        for nombre in _BUILTINS_PERMITIDOS
        if hasattr(_builtins_module, nombre)
    }

    entorno_global = {
        "__builtins__": builtins_restringidos,
        "pd": pd,
        "df": df.copy(),
        "corregir_fecha": _corregir_columna_fecha_robusta,
        "plt": plt,
    }
    entorno_local: dict[str, Any] = {}

    try:
        exec(codigo, entorno_global, entorno_local)

        if "resultado" in entorno_local:
            resultado = entorno_local["resultado"]
        elif "result" in entorno_local:
            resultado = entorno_local["result"]
        else:
            resultado = _NO_ASIGNADO

        if resultado is _NO_ASIGNADO:
            queue.put((
                "error",
                "El código se ejecutó sin errores pero no asignó su resultado a la "
                "variable `resultado` (ni `result`). Agregá esa asignación al final.",
            ))
        else:
            queue.put(("ok", resultado))
    except Exception as e:
        queue.put(("error", str(e)))


def run_sandboxed(codigo: str, df: pd.DataFrame) -> SandboxResult:
    """
    Punto de entrada principal: ejecuta `codigo` (generado por el LLM)
    contra `df`, en un proceso aislado con builtins restringidos y un
    límite de tiempo.

    Convención esperada del código generado: debe asignar su resultado a
    una variable llamada `resultado` (o `result`). Ej.:
        "resultado = df['edad'].mean()"

    Nota importante para código de CORRECCIÓN/ELIMINACIÓN (a diferencia
    de análisis de solo lectura): como acá arriba se le pasa una COPIA
    de `df` al proceso aislado, mutar `df` in-place dentro del código
    generado NO alcanza para que el cambio vuelva al caller -- el código
    debe asignar el DataFrame ya modificado a `resultado` (ej.:
    `resultado = df` al final), o el cambio se pierde cuando termina el
    proceso hijo.

    Devuelve un SandboxResult — nunca lanza una excepción hacia afuera,
    incluso si el código falla, se cuelga, o es rechazado de entrada.
    """
    patron_rechazado = analizar_codigo_estaticamente(codigo)
    if patron_rechazado:
        return SandboxResult(
            success=False,
            error=f"Código rechazado por el análisis estático (patrón prohibido: {patron_rechazado!r}).",
            rejected_by_static_analysis=True,
        )

    queue: multiprocessing.Queue = multiprocessing.Queue()
    proceso = multiprocessing.Process(target=_ejecutar_en_proceso_aislado, args=(codigo, df, queue))
    proceso.start()

    # IMPORTANTE: leer de la queue ANTES de hacer join(), no después.
    # multiprocessing.Queue usa un pipe con buffer limitado por el SO
    # (típicamente ~64KB) -- si el objeto que el hijo pone en la queue
    # (ej. un DataFrame corregido de miles de filas) pesa más que eso,
    # el hijo se bloquea escribiendo hasta que alguien lea del otro
    # lado. Si el padre hace join() ANTES de leer, queda esperando a
    # que el hijo termine, mientras el hijo espera a que el padre lea
    # -- deadlock clásico. Esto causaba el timeout real en producción
    # con datasets de tamaño normal (no era lentitud de cómputo).
    try:
        estado, valor = queue.get(timeout=_TIMEOUT_SEGUNDOS)
    except Exception:
        proceso.terminate()
        proceso.join()
        return SandboxResult(success=False, error=f"El código tardó más de {_TIMEOUT_SEGUNDOS}s y fue detenido.")

    proceso.join(timeout=1)  # ya no debería bloquear -- el put() ya se hizo
    if proceso.is_alive():
        proceso.terminate()
        proceso.join()

    if estado == "ok":
        return SandboxResult(success=True, output=valor)
    return SandboxResult(success=False, error=valor)


if __name__ == "__main__":
    df_ejemplo = pd.DataFrame({"edad": [22, 25, 47, 35, 62]})

    print("--- 1. Código legítimo ---")
    print(run_sandboxed("resultado = df['edad'].mean()", df_ejemplo).to_dict())

    print("\n--- 2. Código malicioso: intenta importar os ---")
    print(run_sandboxed("import os\nresultado = os.listdir('.')", df_ejemplo).to_dict())

    print("\n--- 3. NUEVO: from-import (antes bypaseaba el análisis estático) ---")
    r = run_sandboxed("from os import listdir\nresultado = 1", df_ejemplo)
    print(r.to_dict())

    print("\n--- 4. NUEVO: try/except con tipos de excepción básicos ---")
    print(run_sandboxed(
        "try:\n    resultado = 1/0\nexcept ZeroDivisionError:\n    resultado = -1",
        df_ejemplo,
    ).to_dict())

    print("\n--- 5. NUEVO: isinstance/type ---")
    print(run_sandboxed("resultado = isinstance(df, type(df))", df_ejemplo).to_dict())

    print("\n--- 6. NUEVO: intento de escape vía __class__ -- debe seguir rechazado ---")
    print(run_sandboxed(
        "resultado = ().__class__.__bases__[0].__subclasses__()",
        df_ejemplo,
    ).to_dict())

    print("\n--- 7. Código que se cuelga (loop infinito) ---")
    print(run_sandboxed("while True:\n    pass", df_ejemplo).to_dict())

    print("\n--- 8. Gráfico con plt ya inyectado (sin import) ---")
    resultado_grafico = run_sandboxed(
        "fig, ax = plt.subplots()\nax.bar(df['edad'].index.astype(str), df['edad'])\nresultado = fig",
        df_ejemplo,
    )
    print("success:", resultado_grafico.success)
    print("tipo de output:", type(resultado_grafico.output) if resultado_grafico.success else resultado_grafico.error)