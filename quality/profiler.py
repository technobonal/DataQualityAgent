"""
quality/profiler.py

Motor de calidad determinístico: nulos, duplicados y tipos de dato.
Sin dependencia de LLM — pandas puro, rápido y testeable en aislado.

Los outliers (IQR) viven en quality/outliers.py.
El score final y el reporte consolidado viven en quality/report.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import pandas as pd

# % mínimo de valores no nulos que deben parsear como fecha/número para
# clasificar una columna de texto como "datetime_as_text" / "numeric_as_text".
_DATETIME_TEXT_THRESHOLD = 0.8
_NUMERIC_TEXT_THRESHOLD = 0.8


@dataclass
class ColumnProfile:
    """Perfil de calidad de una columna individual."""
    name: str
    dtype: str
    null_count: int
    null_pct: float
    unique_count: int
    inferred_type: str  # 'numeric', 'numeric_as_text', 'categorical', 'datetime', 'datetime_as_text', 'boolean', 'text', 'empty'
    sample_values: list[Any] = field(default_factory=list)


@dataclass
class DuplicateProfile:
    """Resumen de filas duplicadas."""
    total_duplicates: int
    duplicate_pct: float
    duplicate_indices: list[int]


@dataclass
class DataProfile:
    """Perfil de calidad completo de un DataFrame."""
    row_count: int
    column_count: int
    columns: dict[str, ColumnProfile]
    duplicates: DuplicateProfile
    total_nulls: int
    memory_usage_mb: float

    def to_dict(self) -> dict[str, Any]:
        """Serializa a dict plano — lo que tool_definitions.py le devuelve al LLM."""
        return asdict(self)


def _infer_semantic_type(series: pd.Series) -> str:
    """
    Infiere el tipo semántico de una columna más allá del dtype de pandas.
    Útil porque una columna 'object' puede ser texto libre, categórica,
    o una fecha que Excel guardó como string.
    """
    non_null = series.dropna()
    if non_null.empty:
        return "empty"

    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # A partir de acá el dtype es de texto — puede ser 'object' (pandas <3)
    # o 'str' nativo (pandas >=3, que dejó de usar object para columnas
    # de texto). Hay que chequear ambos casos explícitamente.
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        sample = non_null.head(200)

        # Fechas guardadas como texto: coerce + umbral (no raise), para que
        # una columna de fechas con algunos valores corruptos se siga
        # detectando como fecha — que es justo el problema a señalar.
        parsed_dates = pd.to_datetime(sample, errors="coerce", format="mixed")
        if parsed_dates.notna().mean() >= _DATETIME_TEXT_THRESHOLD:
            return "datetime_as_text"

        # Números guardados como texto (típico: una celda con formato
        # de texto en una columna que debería ser numérica).
        parsed_nums = pd.to_numeric(sample, errors="coerce")
        if parsed_nums.notna().mean() >= _NUMERIC_TEXT_THRESHOLD:
            return "numeric_as_text"

    # Categórica vs texto libre: heurística por cardinalidad relativa
    unique_ratio = non_null.nunique() / len(non_null)
    if unique_ratio < 0.5 and non_null.nunique() < 50:
        return "categorical"

    return "text"


def profile_column(series: pd.Series) -> ColumnProfile:
    """Genera el perfil de calidad de una sola columna."""
    null_count = int(series.isna().sum())
    total = len(series)

    return ColumnProfile(
        name=str(series.name),
        dtype=str(series.dtype),
        null_count=null_count,
        null_pct=round((null_count / total * 100) if total else 0.0, 2),
        unique_count=int(series.nunique(dropna=True)),
        inferred_type=_infer_semantic_type(series),
        sample_values=series.dropna().head(3).tolist(),
    )


def profile_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> DuplicateProfile:
    """
    Detecta filas duplicadas.

    subset: columnas específicas para considerar (ej. solo ['id', 'email']).
            Si es None, compara la fila completa.
    """
    mask = df.duplicated(subset=subset, keep=False)
    duplicate_indices = df.index[mask].tolist()
    total = len(df)

    return DuplicateProfile(
        total_duplicates=len(duplicate_indices),
        duplicate_pct=round((len(duplicate_indices) / total * 100) if total else 0.0, 2),
        duplicate_indices=duplicate_indices,
    )


def profile_dataframe(
    df: pd.DataFrame,
    duplicate_subset: list[str] | None = None,
) -> DataProfile:
    """
    Punto de entrada principal: genera el perfil de calidad completo del DataFrame.

    Uso típico desde tools/tool_definitions.py:
        profile = profile_dataframe(df)
        return profile.to_dict()  # JSON-serializable para el LLM
    """
    columns = {str(col): profile_column(df[col]) for col in df.columns}
    duplicates = profile_duplicates(df, subset=duplicate_subset)
    total_nulls = sum(c.null_count for c in columns.values())

    return DataProfile(
        row_count=len(df),
        column_count=len(df.columns),
        columns=columns,
        duplicates=duplicates,
        total_nulls=total_nulls,
        memory_usage_mb=round(df.memory_usage(deep=True).sum() / (1024 ** 2), 3),
    )


if __name__ == "__main__":
    # Ejemplo rápido para probar el módulo de forma aislada,
    # simulando un Excel con problemas típicos de calidad.
    df_ejemplo = pd.DataFrame({
        "id": [1, 2, 2, 3, None],
        "nombre": ["Ana", "Luis", "Luis", "Marta", "Pedro"],
        "fecha_registro": ["2024-01-15", "2024-01-16", "2024-01-16", "no_es_fecha", "2024-02-01"],
        "monto": [100.5, 200.0, 200.0, None, 50.25],
        "cantidad_como_texto": ["10", "20", "20", "no_aplica", "30"],
    })

    perfil = profile_dataframe(df_ejemplo)

    print(f"Filas: {perfil.row_count} | Columnas: {perfil.column_count}")
    print(f"Nulos totales: {perfil.total_nulls}")
    print(f"Duplicados: {perfil.duplicates.total_duplicates} ({perfil.duplicates.duplicate_pct}%) "
          f"-> índices {perfil.duplicates.duplicate_indices}")
    print()
    for nombre, col in perfil.columns.items():
        print(f"  {nombre:16s} tipo={col.inferred_type:16s} nulos={col.null_pct:5.1f}%  "
              f"únicos={col.unique_count}  ej={col.sample_values}")
