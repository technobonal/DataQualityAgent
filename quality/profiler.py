"""
quality/profiler.py

Motor de calidad determinístico: nulos, duplicados y tipos de dato.
Sin dependencia de LLM — pandas puro, rápido y testeable en aislado.

Los outliers (IQR) viven en quality/outliers.py.
El score final y el reporte consolidado viven en quality/report.py.

MEJORAS:
- "mixed_types": columnas 'object' que mezclan más de un tipo de Python
  (ej. int y str) se detectan y reportan como problema de calidad real,
  en vez de romper la serialización a Arrow en Streamlit en silencio.
- Detección de fechas más estricta: pd.to_datetime(format="mixed") a
  veces interpreta strings que NO son fechas (ej. coordenadas GPS como
  "221711.0620N") como fechas con timezone inválido, disparando un
  FutureWarning y clasificando mal la columna. Se silencia el warning y
  se agrega un sanity-check de año (entre _MIN_YEAR y _MAX_YEAR) para
  descartar esos falsos positivos.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, asdict
from typing import Any

import pandas as pd

# % mínimo de valores no nulos que deben parsear como fecha/número para
# clasificar una columna de texto como "datetime_as_text" / "numeric_as_text".
_DATETIME_TEXT_THRESHOLD = 0.8
_NUMERIC_TEXT_THRESHOLD = 0.8

# Rango de años razonable para considerar una fecha parseada como
# genuina -- descarta falsos positivos donde pandas "adivina" una fecha
# a partir de un string que en realidad es otra cosa (coordenadas,
# códigos, IDs largos, etc.).
_MIN_YEAR = 1900
_MAX_YEAR = 2100


@dataclass
class ColumnProfile:
    """Perfil de calidad de una columna individual."""
    name: str
    dtype: str
    null_count: int
    null_pct: float
    unique_count: int
    inferred_type: str  # 'numeric', 'numeric_as_text', 'categorical', 'datetime', 'datetime_as_text', 'boolean', 'text', 'mixed_types', 'empty'
    sample_values: list[Any] = field(default_factory=list)
    mixed_type_breakdown: dict[str, int] | None = None  # ej. {'str': 8, 'int': 2} -- solo si inferred_type == 'mixed_types'


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


def _detect_mixed_types(series: pd.Series) -> dict[str, int] | None:
    """
    Revisa si una columna 'object' mezcla más de un tipo de Python real
    (ej. int y str en la misma columna) — el caso que rompe la
    serialización a Arrow en Streamlit (pyarrow.lib.ArrowTypeError).

    Devuelve un dict {nombre_tipo: cantidad} si hay mezcla, o None si la
    columna es consistente.
    """
    non_null = series.dropna()
    if non_null.empty:
        return None

    tipos = non_null.map(lambda v: type(v).__name__)
    conteo = tipos.value_counts().to_dict()

    return conteo if len(conteo) > 1 else None


def _parsear_fechas_de_forma_segura(sample: pd.Series) -> pd.Series:
    """
    Intenta parsear una muestra de texto como fechas, con dos capas de
    protección contra falsos positivos:

    1. Silencia el FutureWarning de pandas sobre timezones no
       reconocidos (ej. cuando confunde una coordenada GPS como
       "221711.0620N" con una fecha+timezone "N") -- es ruido de
       consola, no una condición de error real.
    2. Sanity-check de año: cualquier fecha parseada fuera de
       [_MIN_YEAR, _MAX_YEAR] se descarta (se vuelve NaT), porque un
       año fuera de ese rango es casi siempre un falso positivo del
       parser, no una fecha real del dataset.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")

    fuera_de_rango = (parsed.dt.year < _MIN_YEAR) | (parsed.dt.year > _MAX_YEAR)
    parsed[fuera_de_rango.fillna(False)] = pd.NaT

    return parsed


def _infer_semantic_type(series: pd.Series) -> tuple[str, dict[str, int] | None]:
    """
    Infiere el tipo semántico de una columna más allá del dtype de pandas.
    Devuelve (inferred_type, mixed_type_breakdown).

    mixed_types se chequea PRIMERO: una columna con tipos de Python
    mezclados es un problema de calidad en sí mismo, más urgente que
    clasificarla como texto/categórica/etc.
    """
    non_null = series.dropna()
    if non_null.empty:
        return "empty", None

    if pd.api.types.is_bool_dtype(series):
        return "boolean", None
    if pd.api.types.is_numeric_dtype(series):
        return "numeric", None
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime", None

    # Solo las columnas de dtype 'object' pueden mezclar tipos de Python
    # reales; 'str' nativo (pandas >=3) ya garantiza homogeneidad.
    if pd.api.types.is_object_dtype(series):
        mezcla = _detect_mixed_types(series)
        if mezcla:
            return "mixed_types", mezcla

    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        sample = non_null.head(200)

        parsed_dates = _parsear_fechas_de_forma_segura(sample)
        if parsed_dates.notna().mean() >= _DATETIME_TEXT_THRESHOLD:
            return "datetime_as_text", None

        parsed_nums = pd.to_numeric(sample, errors="coerce")
        if parsed_nums.notna().mean() >= _NUMERIC_TEXT_THRESHOLD:
            return "numeric_as_text", None

    unique_ratio = non_null.nunique() / len(non_null)
    if unique_ratio < 0.5 and non_null.nunique() < 50:
        return "categorical", None

    return "text", None


def profile_column(series: pd.Series) -> ColumnProfile:
    """Genera el perfil de calidad de una sola columna."""
    null_count = int(series.isna().sum())
    total = len(series)
    inferred_type, mixed_breakdown = _infer_semantic_type(series)

    return ColumnProfile(
        name=str(series.name),
        dtype=str(series.dtype),
        null_count=null_count,
        null_pct=round((null_count / total * 100) if total else 0.0, 2),
        unique_count=int(series.nunique(dropna=True)),
        inferred_type=inferred_type,
        sample_values=series.dropna().head(3).tolist(),
        mixed_type_breakdown=mixed_breakdown,
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


def make_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve una COPIA del DataFrame lista para mostrar en Streamlit
    (st.dataframe / st.table), convirtiendo a texto solo las columnas
    con tipos mezclados que pyarrow no puede serializar.

    No modifica el DataFrame original ni afecta el pipeline de limpieza
    -- es puramente para la capa de presentación (UI). Usar así en
    ui/*.py, justo antes de mostrar una tabla:

        from quality.profiler import make_arrow_safe
        st.dataframe(make_arrow_safe(df), width="stretch")
    """
    df_seguro = df.copy()
    for col in df_seguro.columns:
        if pd.api.types.is_object_dtype(df_seguro[col]):
            if _detect_mixed_types(df_seguro[col]) is not None:
                df_seguro[col] = df_seguro[col].astype(str)
    return df_seguro


if __name__ == "__main__":
    # Ejemplo rápido para probar el módulo de forma aislada, simulando
    # un Excel con problemas típicos de calidad, incluyendo una columna
    # de tipos mezclados y una de coordenadas GPS (el caso que disparaba
    # el FutureWarning y se clasificaba mal como fecha).
    df_ejemplo = pd.DataFrame({
        "id": [1, 2, 2, 3, None],
        "nombre": ["Ana", "Luis", "Luis", "Marta", "Pedro"],
        "fecha_registro": ["2024-01-15", "2024-01-16", "2024-01-16", "no_es_fecha", "2024-02-01"],
        "monto": [100.5, 200.0, 200.0, None, 50.25],
        "cantidad_como_texto": ["10", "20", "20", "no_aplica", "30"],
        "FederalAgreements": ["EPA", 12, "DOT", 45, "EPA"],
        "coordenada_gps": ["221711.0620N", "222607.0910N", "222212.3400N", "221900.1200N", "222001.5500N"],
    })

    perfil = profile_dataframe(df_ejemplo)

    print(f"Filas: {perfil.row_count} | Columnas: {perfil.column_count}")
    print(f"Nulos totales: {perfil.total_nulls}")
    print(f"Duplicados: {perfil.duplicates.total_duplicates} ({perfil.duplicates.duplicate_pct}%) "
          f"-> índices {perfil.duplicates.duplicate_indices}")
    print()
    for nombre, col in perfil.columns.items():
        extra = f"  ⚠ mezcla={col.mixed_type_breakdown}" if col.mixed_type_breakdown else ""
        print(f"  {nombre:20s} tipo={col.inferred_type:16s} nulos={col.null_pct:5.1f}%  "
              f"únicos={col.unique_count}  ej={col.sample_values}{extra}")

    print("\n--- Vista segura para Streamlit ---")
    print(make_arrow_safe(df_ejemplo).dtypes)