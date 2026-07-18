"""
cleaning/missing_values.py

Identificación y tratamiento de valores faltantes.
Sin dependencia de LLM — pandas puro, testeable en aislado.

Además de los NaN "de verdad" que pandas reconoce, detecta marcadores de
texto que en la práctica también representan un dato faltante (ej. una
celda de Excel con "?", "N/A" o "-" en vez de estar realmente vacía).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import pandas as pd

# Marcadores comunes que representan "faltante" aunque no sean NaN técnicamente.
# Se comparan en minúsculas y sin espacios extra.
_MISSING_MARKERS = {
    "?", "n/a", "na", "null", "none", "-", "--", "s/d", "sin dato",
    "sin datos", "no aplica", "unknown", "desconocido", "nan", "",
}

_ESTRATEGIAS_VALIDAS = (
    "drop_rows", "drop_columns", "mean", "median", "mode", "constant", "ffill", "bfill",
)


@dataclass
class MissingReport:
    """Perfil de valores faltantes de una columna individual."""
    column: str
    null_count: int
    null_pct: float
    marker_count: int = 0  # valores tipo "?", "N/A" que funcionan como nulo pero no lo son técnicamente
    markers_found: list[str] = field(default_factory=list)


@dataclass
class HandleMissingResult:
    """Resumen del tratamiento de valores faltantes aplicado."""
    strategy: str
    columns_processed: list[str] = field(default_factory=list)
    rows_dropped: int = 0
    columns_dropped: list[str] = field(default_factory=list)
    values_filled: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializa a dict plano — lo que tool_definitions.py le devuelve al LLM."""
        return asdict(self)


def identify_missing(df: pd.DataFrame) -> list[MissingReport]:
    """
    Genera un reporte de valores faltantes por columna: nulos reales
    (NaN) + marcadores de texto que actúan como nulo disfrazado.
    """
    total = len(df)
    reportes: list[MissingReport] = []

    for col in df.columns:
        serie = df[col]
        null_count = int(serie.isna().sum())
        marker_count = 0
        markers_found: list[str] = []

        if pd.api.types.is_object_dtype(serie) or pd.api.types.is_string_dtype(serie):
            no_nulos = serie.dropna().astype(str).str.strip().str.lower()
            mask_markers = no_nulos.isin(_MISSING_MARKERS)
            marker_count = int(mask_markers.sum())
            if marker_count:
                markers_found = sorted(no_nulos[mask_markers].unique().tolist())

        reportes.append(
            MissingReport(
                column=str(col),
                null_count=null_count,
                null_pct=round((null_count / total * 100) if total else 0.0, 2),
                marker_count=marker_count,
                markers_found=markers_found,
            )
        )

    return reportes


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str,
    columns: list[str] | None = None,
    constant_value: Any = None,
) -> tuple[pd.DataFrame, HandleMissingResult]:
    """
    Aplica una estrategia de tratamiento de valores faltantes.

    strategy: 'drop_rows'    -> elimina filas con nulos en las columnas indicadas
              'drop_columns' -> elimina columnas indicadas que tengan al menos un nulo
              'mean'/'median'-> rellena con la media/mediana (solo columnas numéricas)
              'mode'         -> rellena con la moda (cualquier tipo de columna)
              'constant'     -> rellena con constant_value
              'ffill'/'bfill'-> propaga el valor anterior/siguiente

    columns: columnas a tratar. Si es None, aplica a todas las columnas del DataFrame.

    Lanza ValueError si strategy no es válida, o si strategy='constant' sin
    pasar constant_value.
    """
    if strategy not in _ESTRATEGIAS_VALIDAS:
        raise ValueError(f"Estrategia desconocida: {strategy!r}. Usar una de {_ESTRATEGIAS_VALIDAS}.")

    df_nuevo = df.copy()
    target_columns = columns if columns is not None else df_nuevo.columns.tolist()
    resultado = HandleMissingResult(strategy=strategy)

    if strategy == "drop_rows":
        columnas_validas = [c for c in target_columns if c in df_nuevo.columns]
        filas_antes = len(df_nuevo)
        df_nuevo = df_nuevo.dropna(subset=columnas_validas)
        resultado.rows_dropped = filas_antes - len(df_nuevo)
        resultado.columns_processed = columnas_validas

    elif strategy == "drop_columns":
        columnas_con_nulos = [
            c for c in target_columns if c in df_nuevo.columns and df_nuevo[c].isna().any()
        ]
        df_nuevo = df_nuevo.drop(columns=columnas_con_nulos)
        resultado.columns_dropped = columnas_con_nulos

    elif strategy in ("mean", "median"):
        for col in target_columns:
            if col not in df_nuevo.columns or not pd.api.types.is_numeric_dtype(df_nuevo[col]):
                continue
            n_nulos = int(df_nuevo[col].isna().sum())
            if n_nulos == 0:
                continue
            valor = df_nuevo[col].mean() if strategy == "mean" else df_nuevo[col].median()
            df_nuevo[col] = df_nuevo[col].fillna(valor)
            resultado.values_filled[col] = n_nulos
            resultado.columns_processed.append(col)

    elif strategy == "mode":
        for col in target_columns:
            if col not in df_nuevo.columns:
                continue
            n_nulos = int(df_nuevo[col].isna().sum())
            if n_nulos == 0:
                continue
            moda = df_nuevo[col].mode(dropna=True)
            if moda.empty:
                continue
            df_nuevo[col] = df_nuevo[col].fillna(moda.iloc[0])
            resultado.values_filled[col] = n_nulos
            resultado.columns_processed.append(col)

    elif strategy == "constant":
        if constant_value is None:
            raise ValueError("strategy='constant' requiere pasar constant_value")
        for col in target_columns:
            if col not in df_nuevo.columns:
                continue
            n_nulos = int(df_nuevo[col].isna().sum())
            if n_nulos == 0:
                continue
            df_nuevo[col] = df_nuevo[col].fillna(constant_value)
            resultado.values_filled[col] = n_nulos
            resultado.columns_processed.append(col)

    else:  # ffill / bfill
        for col in target_columns:
            if col not in df_nuevo.columns:
                continue
            n_nulos_antes = int(df_nuevo[col].isna().sum())
            if n_nulos_antes == 0:
                continue
            df_nuevo[col] = df_nuevo[col].ffill() if strategy == "ffill" else df_nuevo[col].bfill()
            n_nulos_despues = int(df_nuevo[col].isna().sum())
            resultado.values_filled[col] = n_nulos_antes - n_nulos_despues
            resultado.columns_processed.append(col)

    return df_nuevo, resultado


if __name__ == "__main__":
    # Ejemplo rápido para probar el módulo de forma aislada.
    df_ejemplo = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "edad": [22, None, 47, 35, None],
        "ciudad": ["SP", "?", "RJ", "N/A", "SP"],
    })

    print("--- identify_missing ---")
    for r in identify_missing(df_ejemplo):
        print(r)

    print("\n--- handle_missing_values(strategy='mean') ---")
    df_res, resumen = handle_missing_values(df_ejemplo, strategy="mean", columns=["edad"])
    print(df_res)
    print(resumen.to_dict())

    print("\n--- handle_missing_values(strategy='drop_rows') ---")
    df_res2, resumen2 = handle_missing_values(df_ejemplo, strategy="drop_rows")
    print(df_res2)
    print(resumen2.to_dict())
    