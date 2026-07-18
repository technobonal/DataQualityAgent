"""
quality/outliers.py

Detección y tratamiento de valores atípicos (outliers) en columnas
numéricas, usando el método IQR (rango intercuartílico) — el estándar
para esto, robusto frente a distribuciones no normales.
Sin dependencia de LLM — pandas puro, testeable en aislado.

El perfil de nulos/duplicados/tipos vive en quality/profiler.py.
El reporte consolidado (perfil + outliers + score) vive en quality/report.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import pandas as pd

_ESTRATEGIAS_VALIDAS = ("remove", "cap")


@dataclass
class OutlierReport:
    """Perfil de outliers de una columna individual."""
    column: str
    outlier_count: int
    outlier_pct: float
    lower_bound: float
    upper_bound: float
    sample_outlier_values: list[Any] = field(default_factory=list)


@dataclass
class HandleOutliersResult:
    """Resumen del tratamiento de outliers aplicado."""
    strategy: str
    columns_processed: list[str] = field(default_factory=list)
    rows_removed: int = 0
    values_capped: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializa a dict plano — lo que tool_definitions.py le devuelve al LLM."""
        return asdict(self)


def _iqr_bounds(serie: pd.Series, multiplier: float) -> tuple[float, float]:
    """Calcula los límites inferior/superior del método IQR para una serie."""
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - multiplier * iqr), float(q3 + multiplier * iqr)


def detect_outliers_iqr(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    multiplier: float = 1.5,
) -> list[OutlierReport]:
    """
    Detecta outliers en columnas numéricas con el método IQR: cualquier
    valor fuera de [Q1 - multiplier*IQR, Q3 + multiplier*IQR] se marca
    como atípico.

    columns:    columnas a revisar. Si es None, revisa todas las numéricas.
    multiplier: qué tan estricto es el criterio. 1.5 es el estándar
                (Tukey); usar 3.0 para detectar solo outliers "extremos".

    Columnas no numéricas o no encontradas se saltean silenciosamente
    (no tiene sentido buscar outliers en texto).
    """
    target_columns = columns if columns is not None else df.select_dtypes(include="number").columns.tolist()
    reportes: list[OutlierReport] = []

    for col in target_columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue

        serie = df[col]
        total = int(serie.notna().sum())
        if total == 0:
            continue

        lower, upper = _iqr_bounds(serie, multiplier)
        mask = (serie < lower) | (serie > upper)
        outlier_count = int(mask.sum())

        reportes.append(
            OutlierReport(
                column=str(col),
                outlier_count=outlier_count,
                outlier_pct=round((outlier_count / total * 100) if total else 0.0, 2),
                lower_bound=round(lower, 4),
                upper_bound=round(upper, 4),
                sample_outlier_values=serie[mask].dropna().head(5).tolist(),
            )
        )

    return reportes


def handle_outliers(
    df: pd.DataFrame,
    strategy: str = "cap",
    columns: list[str] | None = None,
    multiplier: float = 1.5,
) -> tuple[pd.DataFrame, HandleOutliersResult]:
    """
    Aplica una estrategia de tratamiento a los outliers detectados por IQR.

    strategy: 'cap'    -> recorta (winsoriza) los valores fuera de rango
                          al límite inferior/superior, sin perder filas.
              'remove' -> elimina las filas que tengan un outlier en
                          cualquiera de las columnas indicadas.
    columns:  columnas a tratar. Si es None, aplica a todas las numéricas.
    multiplier: mismo criterio que detect_outliers_iqr.

    Lanza ValueError si strategy no es 'cap' ni 'remove'.
    """
    if strategy not in _ESTRATEGIAS_VALIDAS:
        raise ValueError(f"Estrategia desconocida: {strategy!r}. Usar una de {_ESTRATEGIAS_VALIDAS}.")

    df_nuevo = df.copy()
    target_columns = columns if columns is not None else df_nuevo.select_dtypes(include="number").columns.tolist()
    resultado = HandleOutliersResult(strategy=strategy)

    if strategy == "cap":
        for col in target_columns:
            if col not in df_nuevo.columns or not pd.api.types.is_numeric_dtype(df_nuevo[col]):
                continue
            lower, upper = _iqr_bounds(df_nuevo[col], multiplier)
            mask = (df_nuevo[col] < lower) | (df_nuevo[col] > upper)
            n_outliers = int(mask.sum())
            if n_outliers == 0:
                continue
            df_nuevo[col] = df_nuevo[col].clip(lower=lower, upper=upper)
            resultado.values_capped[col] = n_outliers
            resultado.columns_processed.append(col)

    else:  # remove
        mask_total = pd.Series(False, index=df_nuevo.index)
        for col in target_columns:
            if col not in df_nuevo.columns or not pd.api.types.is_numeric_dtype(df_nuevo[col]):
                continue
            lower, upper = _iqr_bounds(df_nuevo[col], multiplier)
            mask_col = (df_nuevo[col] < lower) | (df_nuevo[col] > upper)
            mask_total = mask_total | mask_col.fillna(False)
            resultado.columns_processed.append(col)

        filas_antes = len(df_nuevo)
        df_nuevo = df_nuevo[~mask_total]
        resultado.rows_removed = filas_antes - len(df_nuevo)

    return df_nuevo, resultado


if __name__ == "__main__":
    # Ejemplo rápido para probar el módulo de forma aislada.
    df_ejemplo = pd.DataFrame({
        "edad": [22, 25, 47, 35, 62, 300],       # 300 es un outlier evidente
        "salario": [3000, 3500, 8000, 5000, 12000, 11500],
    })

    print("--- detect_outliers_iqr ---")
    for r in detect_outliers_iqr(df_ejemplo):
        print(r)

    print("\n--- handle_outliers(strategy='cap') ---")
    df_res, resumen = handle_outliers(df_ejemplo, strategy="cap", columns=["edad"])
    print(df_res)
    print(resumen.to_dict())

    print("\n--- handle_outliers(strategy='remove') ---")
    df_res2, resumen2 = handle_outliers(df_ejemplo, strategy="remove", columns=["edad"])
    print(df_res2)
    print(resumen2.to_dict())