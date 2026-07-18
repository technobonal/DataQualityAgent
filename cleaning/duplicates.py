"""
cleaning/duplicates.py

Identificación y tratamiento de filas duplicadas.
Sin dependencia de LLM — pandas puro, testeable en aislado.

Complementa a quality/profiler.py (que ya detecta duplicados como parte
del diagnóstico general) con la acción concreta para resolverlos.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import pandas as pd

_ESTRATEGIAS_VALIDAS = ("keep_first", "keep_last", "remove_all")


@dataclass
class DuplicateReport:
    """Perfil de filas duplicadas en el DataFrame."""
    total_duplicates: int
    duplicate_pct: float
    duplicate_indices: list[int] = field(default_factory=list)

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


def identify_duplicates(
    df: pd.DataFrame,
    subset: list[str] | None = None,
) -> DuplicateReport:
    """
    Detecta filas duplicadas sin modificar el DataFrame.

    subset: columnas específicas para considerar (ej. solo ['id', 'email']).
            Si es None, compara la fila completa.
    """
    mask = df.duplicated(subset=subset, keep=False)
    duplicate_indices = df.index[mask].tolist()
    total = len(df)

    return DuplicateReport(
        total_duplicates=len(duplicate_indices),
        duplicate_pct=round((len(duplicate_indices) / total * 100) if total else 0.0, 2),
        duplicate_indices=duplicate_indices,
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
    # Ejemplo rápido para probar el módulo de forma aislada.
    df_ejemplo = pd.DataFrame({
        "id": [1, 2, 2, 3, 4, 4],
        "nombre": ["Ana", "Luis", "Luis", "Marta", "Pedro", "Pedro Distinto"],
    })

    print("--- identify_duplicates (fila completa) ---")
    print(identify_duplicates(df_ejemplo).to_dict())

    print("\n--- identify_duplicates (subset=['id']) ---")
    print(identify_duplicates(df_ejemplo, subset=["id"]).to_dict())

    print("\n--- handle_duplicates(strategy='keep_first', subset=['id']) ---")
    df_res, resumen = handle_duplicates(df_ejemplo, strategy="keep_first", subset=["id"])
    print(df_res)
    print(resumen.to_dict())

    print("\n--- handle_duplicates(strategy='remove_all', subset=['id']) ---")
    df_res2, resumen2 = handle_duplicates(df_ejemplo, strategy="remove_all", subset=["id"])
    print(df_res2)
    print(resumen2.to_dict())