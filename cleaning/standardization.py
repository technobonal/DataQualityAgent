"""
cleaning/standardization.py

Estandarización (z-score) de columnas numéricas: transforma cada valor a
cuántos desvíos estándar está de la media (media resultante = 0, desvío = 1).
Sin dependencia de LLM — pandas puro, testeable en aislado.

Útil antes de entrenar modelos sensibles a la escala (regresión lineal,
KNN, SVM, redes neuronales) o para comparar columnas con unidades distintas
(ej. edad vs. salario).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import pandas as pd


@dataclass
class StandardizationResult:
    """Resumen de la estandarización aplicada al DataFrame."""
    columns_processed: list[str] = field(default_factory=list)
    columns_skipped: list[str] = field(default_factory=list)  # no numéricas o desvío == 0
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializa a dict plano — lo que tool_definitions.py le devuelve al LLM."""
        return asdict(self)


def standardize_columns(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    suffix: str | None = None,
) -> tuple[pd.DataFrame, StandardizationResult]:
    """
    Aplica estandarización z-score: (x - media) / desvío_estándar.

    columns: columnas a estandarizar. Si es None, aplica a todas las numéricas
              del DataFrame.
    suffix:  si se pasa (ej. "_z"), crea columnas nuevas con ese sufijo en vez
              de sobrescribir las originales. Si es None, sobrescribe in place.

    Columnas no numéricas o con desvío estándar 0 (todos los valores iguales,
    lo que produciría división por cero) se saltean y quedan registradas en
    columns_skipped en lugar de romper la ejecución.
    """
    df_nuevo = df.copy()
    target_columns = (
        columns if columns is not None else df.select_dtypes(include="number").columns.tolist()
    )

    resultado = StandardizationResult()

    for col in target_columns:
        if col not in df_nuevo.columns or not pd.api.types.is_numeric_dtype(df_nuevo[col]):
            resultado.columns_skipped.append(col)
            continue

        media = df_nuevo[col].mean()
        desvio = df_nuevo[col].std()

        if pd.isna(desvio) or desvio == 0:
            resultado.columns_skipped.append(col)
            continue

        valores_estandarizados = (df_nuevo[col] - media) / desvio
        destino = f"{col}{suffix}" if suffix else col
        df_nuevo[destino] = valores_estandarizados

        resultado.columns_processed.append(destino)
        resultado.means[col] = round(float(media), 4)
        resultado.stds[col] = round(float(desvio), 4)

    return df_nuevo, resultado


if __name__ == "__main__":
    # Ejemplo rápido para probar el módulo de forma aislada.
    df_ejemplo = pd.DataFrame({
        "edad": [22, 25, 47, 35, 62],
        "salario": [3000, 3500, 8000, 5000, 12000],
        "categoria": ["A", "B", "A", "C", "B"],
        "constante": [1, 1, 1, 1, 1],  # para probar el caso desvío == 0
    })

    df_resultado, resumen = standardize_columns(df_ejemplo, columns=["edad", "salario", "categoria", "constante"], suffix="_z")
    print(df_resultado)
    print()
    print(resumen.to_dict())