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
"""

from __future__ import annotations

import builtins as _builtins_module
import multiprocessing
import re
from dataclasses import dataclass
from typing import Any

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
    r"__import__\s*\(", r"\bopen\s*\(", r"\beval\s*\(", r"\bexec\s*\(",
    r"\bcompile\s*\(", r"\bgetattr\s*\(", r"\bsetattr\s*\(", r"\bdelattr\s*\(",
    r"\bglobals\s*\(", r"\blocals\s*\(", r"\bvars\s*\(",
    r"__[a-zA-Z_]+__",  # cualquier acceso a dunder attributes/métodos
    r"\bos\.", r"\bsys\.", r"\bsubprocess\.",
]

# Builtins mínimos necesarios para operaciones normales de pandas/cálculo.
# Todo lo que no está en esta lista (open, __import__, eval, exec, input,
# etc.) directamente no existe dentro del código ejecutado.
_BUILTINS_PERMITIDOS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "int", "len", "list", "map", "max", "min", "print", "range", "reversed",
    "round", "set", "sorted", "str", "sum", "tuple", "zip",
    "True", "False", "None",
}


def _corregir_columna_fecha_robusta(serie: pd.Series) -> pd.Series:
    """
    Convierte una columna de texto a datetime.

    HISTORIA DEL DISEÑO (dos iteraciones, ver security/sandbox.py __main__
    para los benchmarks):

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

    # Formatos comunes a probar, en orden -- se puede ampliar esta lista
    # si en tu dataset real aparecen formatos que no están cubiertos acá.
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

    # Fallback lento -- SOLO sobre lo que quedó pendiente tras probar
    # todos los formatos comunes (en la práctica, un subconjunto chico).
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
        # Función pre-testeada para columnas de fecha -- ver
        # _corregir_columna_fecha_robusta arriba para el motivo. El
        # código generado la puede llamar como corregir_fecha(df['col']).
        "corregir_fecha": _corregir_columna_fecha_robusta,
    }
    entorno_local: dict[str, Any] = {}

    try:
        exec(codigo, entorno_global, entorno_local)

        # FIX: antes se hacía entorno_local.get("resultado", entorno_local.get("result")),
        # que devuelve None tanto si la variable no existe como si existe y vale None.
        # Ahora se busca explícitamente CUÁL de las dos claves está presente, para
        # poder avisar si NINGUNA fue asignada (en vez de reportarlo como éxito).
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

    Nota importante para código de CORRECCIÓN (a diferencia de análisis
    de solo lectura): como acá arriba se le pasa una COPIA de `df` al
    proceso aislado, mutar `df` in-place dentro del código generado NO
    alcanza para que el cambio vuelva al caller -- el código debe
    asignar el DataFrame ya corregido a `resultado` (ej.: `resultado =
    df` al final), o el cambio se pierde cuando termina el proceso hijo.

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
    # -- deadlock clásico, documentado en la letra chica de la librería
    # estándar. Esto causaba el timeout real en producción con datasets
    # de tamaño normal (no era lentitud de cómputo -- confirmado con
    # benchmarks, ver _corregir_columna_fecha_robusta).
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

    print("\n--- 3. Código malicioso: intenta leer archivos del sistema ---")
    print(run_sandboxed("resultado = open('/etc/passwd').read()", df_ejemplo).to_dict())

    print("\n--- 4. Código que se cuelga (loop infinito) ---")
    print(run_sandboxed("while True:\n    pass", df_ejemplo).to_dict())

    print("\n--- 5. Código con error normal de ejecución (división por cero) ---")
    print(run_sandboxed("resultado = 1/0", df_ejemplo).to_dict())

    print("\n--- 6. NUEVO: código que corre bien pero NO asigna resultado (el bug del None) ---")
    print(run_sandboxed("x = df['edad'].mean()\nprint(x)", df_ejemplo).to_dict())

    print("\n--- 7. NUEVO: código que corrige un dtype numérico y devuelve el df corregido ---")
    df_texto = pd.DataFrame({"edad": ["22", "25", "47"]})
    resultado_correccion = run_sandboxed(
        "df['edad'] = pd.to_numeric(df['edad'], errors='coerce')\nresultado = df",
        df_texto,
    )
    print("success:", resultado_correccion.success)
    print("dtypes corregidos:\n", resultado_correccion.output.dtypes if resultado_correccion.success else resultado_correccion.error)

    print("\n--- 8. NUEVO: corregir_fecha() con formatos MEZCLADOS en la misma columna ---")
    df_fechas_mezcladas = pd.DataFrame({
        "fecha": ["2024-01-15", "15/03/2024", "March 20, 2024", "2024.05.10", "sin fecha"],
    })
    resultado_fecha = run_sandboxed(
        "df['fecha'] = corregir_fecha(df['fecha'])\nresultado = df",
        df_fechas_mezcladas,
    )
    print("success:", resultado_fecha.success)
    if resultado_fecha.success:
        print(resultado_fecha.output)
        print("dtype:", resultado_fecha.output["fecha"].dtype)
        print("Se recuperaron 4 de 5 fechas válidas (esperado):", resultado_fecha.output["fecha"].notna().sum() == 4)
    else:
        print("error:", resultado_fecha.error)