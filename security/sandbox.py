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

    entorno_global = {"__builtins__": builtins_restringidos, "pd": pd, "df": df.copy()}
    entorno_local: dict[str, Any] = {}

    try:
        exec(codigo, entorno_global, entorno_local)
        resultado = entorno_local.get("resultado", entorno_local.get("result"))
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
    proceso.join(timeout=_TIMEOUT_SEGUNDOS)

    if proceso.is_alive():
        proceso.terminate()
        proceso.join()
        return SandboxResult(success=False, error=f"El código tardó más de {_TIMEOUT_SEGUNDOS}s y fue detenido.")

    if queue.empty():
        return SandboxResult(success=False, error="El proceso terminó sin devolver resultado (pudo haber crasheado).")

    estado, valor = queue.get()
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