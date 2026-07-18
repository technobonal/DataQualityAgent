"""
cleaning/binning.py

Binning (discretización) de una columna numérica en rangos/categorías.
Sin dependencia de LLM — pandas puro (usa pd.cut / pd.qcut).

Útil para convertir una variable continua (ej. edad, salario) en
categorías (ej. "joven / adulto / mayor") para análisis exploratorio o
para modelos que funcionan mejor con variables categóricas.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import pandas as pd

_METODOS_VALIDOS = ("equal_width", "equal_frequency", "custom")


@dataclass
class BinningResult:
    """Resumen del binning aplicado a una columna."""
    column: str
    new_column: str
    method: str  # 'equal_width', 'equal_frequency' o 'custom'
    bin_count: int
    bin_edges: list[float] = field(default_factory=list)
    bin_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializa a dict plano — lo que tool_definitions.py le devuelve al LLM."""
        return asdict(self)


def bin_column(
    df: pd.DataFrame,
    column: str,
    method: str = "equal_width",
    bins: int | list[float] = 5,
    labels: list[str] | None = None,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, BinningResult]:
    """
    Discretiza una columna numérica en bins/categorías, agregando una
    columna nueva (no sobrescribe la original — a diferencia de
    standardization/normalization, acá siempre conviene conservar el
    valor numérico de origen).

    column: columna numérica a discretizar.
    method: 'equal_width'     -> rangos de igual ancho (pd.cut)
            'equal_frequency' -> misma cantidad de filas por bin (pd.qcut)
            'custom'          -> bins ya define los límites exactos
    bins:   cantidad de bins (int) para equal_width/equal_frequency,
            o lista de límites (list[float]) cuando method='custom'.
    labels: nombres para cada bin (ej. ["bajo", "medio", "alto"]). Si se
            omite, pandas genera etiquetas tipo "(20.0, 35.0]".
    new_column: nombre de la columna resultado. Si se omite, usa
                "{column}_bin".

    Lanza ValueError si la columna no existe, no es numérica, o si
    method/bins no son consistentes entre sí.
    """
    if column not in df.columns:
        raise ValueError(f"Columna no encontrada: {column!r}")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"La columna {column!r} no es numérica")
    if method not in _METODOS_VALIDOS:
        raise ValueError(f"Método desconocido: {method!r}. Usar uno de {_METODOS_VALIDOS}.")

    df_nuevo = df.copy()
    destino = new_column or f"{column}_bin"

    if method == "custom":
        if not isinstance(bins, list):
            raise ValueError("Para method='custom', bins debe ser una lista de límites")
        categorias, edges = pd.cut(df_nuevo[column], bins=bins, labels=labels, retbins=True)
    elif method == "equal_width":
        if not isinstance(bins, int):
            raise ValueError("Para method='equal_width', bins debe ser un entero (cantidad de bins)")
        categorias, edges = pd.cut(df_nuevo[column], bins=bins, labels=labels, retbins=True)
    else:  # equal_frequency
        if not isinstance(bins, int):
            raise ValueError("Para method='equal_frequency', bins debe ser un entero (cantidad de bins)")
        categorias, edges = pd.qcut(
            df_nuevo[column], q=bins, labels=labels, retbins=True, duplicates="drop"
        )

    df_nuevo[destino] = categorias

    # bin_count real: si labels vino con nombres custom, categories no es
    # numérico y el largo de labels es la fuente de verdad; si no, contamos
    # las categorías efectivamente generadas (equal_frequency puede colapsar
    # bins con duplicates="drop").
    conteo_bins = len(labels) if labels is not None else int(categorias.cat.categories.size)

    resultado = BinningResult(
        column=column,
        new_column=destino,
        method=method,
        bin_count=conteo_bins,
        bin_edges=[round(float(e), 4) for e in edges],
        bin_counts={str(k): int(v) for k, v in categorias.value_counts().sort_index().items()},
    )

    return df_nuevo, resultado


if __name__ == "__main__":
    # Ejemplo rápido para probar el módulo de forma aislada.
    df_ejemplo = pd.DataFrame({
        "edad": [18, 22, 25, 31, 40, 47, 55, 62, 70, 81],
    })

    print("--- equal_width (4 bins) ---")
    df_res, resumen = bin_column(df_ejemplo, "edad", method="equal_width", bins=4)
    print(df_res)
    print(resumen.to_dict())

    print("\n--- equal_frequency (3 bins) ---")
    df_res2, resumen2 = bin_column(df_ejemplo, "edad", method="equal_frequency", bins=3)
    print(df_res2)
    print(resumen2.to_dict())

    print("\n--- custom (límites propios + labels) ---")
    df_res3, resumen3 = bin_column(
        df_ejemplo, "edad", method="custom",
        bins=[0, 30, 60, 100], labels=["joven", "adulto", "mayor"],
    )
    print(df_res3)
    print(resumen3.to_dict())