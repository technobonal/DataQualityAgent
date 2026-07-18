"""
cleaning/normalization.py

Normalización min-max de columnas numéricas: reescala los valores a un
rango fijo (por defecto [0, 1]). Sin dependencia de LLM — pandas puro.

Complementa a cleaning/standardization.py (z-score): acá se usa cuando
se prefiere un rango acotado y conocido de antemano en vez de media/desvío
(por ejemplo para redes neuronales que esperan entradas en [0, 1]).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import pandas as pd


@dataclass
class NormalizationResult:
    """Resumen de la normalización aplicada al DataFrame."""
    columns_processed: list[str] = field(default_factory=list)
    columns_skipped: list[str] = field(default_factory=list)  # no numéricas o min == max
    mins: dict[str, float] = field(default_factory=dict)
    maxs: dict[str, float] = field(default_factory=dict)
    range_min: float = 0.0
    range_max: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serializa a dict plano — lo que tool_definitions.py le devuelve al LLM."""
        return asdict(self)


def normalize_columns(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    range_min: float = 0.0,
    range_max: float = 1.0,
    suffix: str | None = None,
) -> tuple[pd.DataFrame, NormalizationResult]:
    """
    Aplica normalización min-max, reescalando a [range_min, range_max].

    columns: columnas a normalizar. Si es None, aplica a todas las numéricas
              del DataFrame.
    suffix:  si se pasa (ej. "_norm"), crea columnas nuevas con ese sufijo en
              vez de sobrescribir las originales. Si es None, sobrescribe in place.

    Columnas no numéricas o constantes (min == max, lo que produciría
    división por cero) se saltean y quedan registradas en columns_skipped
    en lugar de romper la ejecución.

    Lanza ValueError si range_min >= range_max (rango inválido).
    """
    if range_min >= range_max:
        raise ValueError("range_min debe ser menor que range_max")

    df_nuevo = df.copy()
    target_columns = (
        columns if columns is not None else df.select_dtypes(include="number").columns.tolist()
    )

    resultado = NormalizationResult(range_min=range_min, range_max=range_max)

    for col in target_columns:
        if col not in df_nuevo.columns or not pd.api.types.is_numeric_dtype(df_nuevo[col]):
            resultado.columns_skipped.append(col)
            continue

        minimo = df_nuevo[col].min()
        maximo = df_nuevo[col].max()

        if pd.isna(minimo) or pd.isna(maximo) or maximo == minimo:
            resultado.columns_skipped.append(col)
            continue

        escala = (df_nuevo[col] - minimo) / (maximo - minimo)
        valores_normalizados = escala * (range_max - range_min) + range_min

        destino = f"{col}{suffix}" if suffix else col
        df_nuevo[destino] = valores_normalizados

        resultado.columns_processed.append(destino)
        resultado.mins[col] = round(float(minimo), 4)
        resultado.maxs[col] = round(float(maximo), 4)

    return df_nuevo, resultado


if __name__ == "__main__":
    # Ejemplo rápido para probar el módulo de forma aislada.
    df_ejemplo = pd.DataFrame({
        "edad": [22, 25, 47, 35, 62],
        "salario": [3000, 3500, 8000, 5000, 12000],
        "categoria": ["A", "B", "A", "C", "B"],
        "constante": [1, 1, 1, 1, 1],  # para probar el caso min == max
    })

    df_resultado, resumen = normalize_columns(df_ejemplo, columns=["edad", "salario", "categoria", "constante"], suffix="_norm")
    print(df_resultado)
    print()
    print(resumen.to_dict())