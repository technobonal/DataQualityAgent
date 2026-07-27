"""
cleaning/duplicates.py

Identificación y tratamiento de filas duplicadas.
Sin dependencia de LLM — pandas puro, testeable en aislado.

Complementa a quality/profiler.py (que ya detecta duplicados como parte
del diagnóstico general) con la acción concreta para resolverlos.

MEJORA: identify_duplicates() ahora además agrupa los duplicados por el
valor que se repite (ej. todos los id=2 juntos), en vez de devolver
únicamente una lista plana de índices de fila sin contexto.
duplicate_indices se mantiene igual (lo sigue usando ui/chat_tab.py para
armar la tabla de duplicados) -- duplicate_groups es un campo nuevo,
pensado para que el LLM pueda explicar CUÁL valor se repite y CUÁNTAS
veces, en vez de solo ver números de fila sueltos. Los grupos vienen
ordenados de mayor a menor cantidad de repeticiones, mismo criterio de
relevancia que ya aplicamos en missing_values.py y outliers.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import pandas as pd

_ESTRATEGIAS_VALIDAS = ("keep_first", "keep_last", "remove_all")

# Cuántos índices de ejemplo se guardan por grupo -- no hace falta la
# lista completa para que el LLM entienda el patrón (para eso ya está
# duplicate_indices con la lista completa, para la UI).
_MUESTRA_INDICES_POR_GRUPO = 5


@dataclass
class DuplicateGroup:
    """Un conjunto de filas que comparten el mismo valor duplicado."""
    values: dict[str, Any]  # columna -> valor que se repite (o combinación, si subset tiene varias columnas)
    count: int              # cuántas filas comparten ese valor
    sample_indices: list[int] = field(default_factory=list)  # primeros N índices de fila, para referencia


@dataclass
class DuplicateReport:
    """Perfil de filas duplicadas en el DataFrame."""
    total_duplicates: int
    duplicate_pct: float
    duplicate_indices: list[int] = field(default_factory=list)
    duplicate_groups: list[DuplicateGroup] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serializa a dict plano — lo que tool_definitions.py le devuelve al LLM."""
        return asdict(self)


@dataclass
class HandleDuplicatesResult:
    """Resumen del tratamiento de duplicados aplicado."""
    strategy: str
    subset: list[str] | None = None
    rows_removed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serializa a dict plano — lo que tool_definitions.py le devuelve al LLM."""
        return asdict(self)


def _valor_serializable(v: Any) -> Any:
    """Convierte tipos numpy/pandas a tipos nativos de Python para que json.dumps no falle."""
    if pd.isna(v):
        return None
    if hasattr(v, "item"):  # numpy scalar (int64, float64, etc.)
        return v.item()
    return v


def identify_duplicates(
    df: pd.DataFrame,
    subset: list[str] | None = None,
    min_group_count: int | None = None,
) -> DuplicateReport:
    """
    Detecta filas duplicadas sin modificar el DataFrame.

    subset:          columnas específicas para considerar (ej. solo
                      ['id', 'email']). Si es None, compara la fila completa.
    min_group_count: si se pasa, en duplicate_groups devuelve SOLO los
                      grupos con count >= ese valor -- para pedidos tipo
                      "qué valores se repiten más de 3 veces".

    duplicate_groups viene SIEMPRE ordenado de mayor a menor cantidad de
    repeticiones, para que los patrones más frecuentes queden primero
    (relevante porque el LLM solo ve un máximo de elementos por lista --
    ver _recortar_para_llm en core/agent.py).
    """
    mask = df.duplicated(subset=subset, keep=False)
    filas_duplicadas = df[mask]
    duplicate_indices = df.index[mask].tolist()
    total = len(df)

    columnas_agrupar = subset if subset else df.columns.tolist()
    grupos: list[DuplicateGroup] = []

    if not filas_duplicadas.empty:
        agrupado = filas_duplicadas.groupby(columnas_agrupar, dropna=False)
        for clave, grupo_df in agrupado:
            if not isinstance(clave, tuple):
                clave = (clave,)
            valores = {
                str(col): _valor_serializable(val)
                for col, val in zip(columnas_agrupar, clave)
            }
            grupos.append(DuplicateGroup(
                values=valores,
                count=len(grupo_df),
                sample_indices=grupo_df.index.tolist()[:_MUESTRA_INDICES_POR_GRUPO],
            ))

    grupos.sort(key=lambda g: g.count, reverse=True)

    if min_group_count is not None:
        grupos = [g for g in grupos if g.count >= min_group_count]

    return DuplicateReport(
        total_duplicates=len(duplicate_indices),
        duplicate_pct=round((len(duplicate_indices) / total * 100) if total else 0.0, 2),
        duplicate_indices=duplicate_indices,
        duplicate_groups=grupos,
    )


def handle_duplicates(
    df: pd.DataFrame,
    strategy: str = "keep_first",
    subset: list[str] | None = None,
) -> tuple[pd.DataFrame, HandleDuplicatesResult]:
    """
    Elimina filas duplicadas según la estrategia elegida.

    strategy: 'keep_first'  -> conserva la primera aparición de cada duplicado
              'keep_last'   -> conserva la última aparición
              'remove_all'  -> elimina TODAS las filas que tengan algún
                               duplicado (no conserva ninguna copia)
    subset: columnas específicas para considerar. Si es None, compara la
            fila completa.

    Lanza ValueError si strategy no es una de las 3 válidas.
    """
    if strategy not in _ESTRATEGIAS_VALIDAS:
        raise ValueError(f"Estrategia desconocida: {strategy!r}. Usar una de {_ESTRATEGIAS_VALIDAS}.")

    df_nuevo = df.copy()
    filas_antes = len(df_nuevo)

    if strategy == "keep_first":
        df_nuevo = df_nuevo.drop_duplicates(subset=subset, keep="first")
    elif strategy == "keep_last":
        df_nuevo = df_nuevo.drop_duplicates(subset=subset, keep="last")
    else:  # remove_all
        df_nuevo = df_nuevo.drop_duplicates(subset=subset, keep=False)

    resultado = HandleDuplicatesResult(
        strategy=strategy,
        subset=subset,
        rows_removed=filas_antes - len(df_nuevo),
    )

    return df_nuevo, resultado


if __name__ == "__main__":
    # Ejemplo con un valor que se repite MUCHO más que los demás, para
    # confirmar que queda primero en duplicate_groups.
    df_ejemplo = pd.DataFrame({
        "id": [1, 2, 2, 3, 4, 4, 5, 5, 5, 5],
        "nombre": ["Ana", "Luis", "Luis", "Marta", "Pedro", "Pedro Distinto",
                   "Sofia", "Sofia", "Sofia", "Sofia"],
    })

    print("--- identify_duplicates (fila completa) ---")
    reporte = identify_duplicates(df_ejemplo)
    print("total_duplicates:", reporte.total_duplicates)
    print("duplicate_groups:")
    for g in reporte.duplicate_groups:
        print(f"  {g.values} -> count={g.count} sample_indices={g.sample_indices}")

    print("\n--- identify_duplicates (subset=['id']) ---")
    reporte2 = identify_duplicates(df_ejemplo, subset=["id"])
    for g in reporte2.duplicate_groups:
        print(f"  {g.values} -> count={g.count}")

    print("\n--- identify_duplicates (subset=['id'], min_group_count=3) ---")
    reporte3 = identify_duplicates(df_ejemplo, subset=["id"], min_group_count=3)
    for g in reporte3.duplicate_groups:
        print(f"  {g.values} -> count={g.count}")

    print("\n--- handle_duplicates(strategy='keep_first', subset=['id']) sigue igual ---")
    df_res, resumen = handle_duplicates(df_ejemplo, strategy="keep_first", subset=["id"])
    print(resumen.to_dict())