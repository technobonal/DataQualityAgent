"""
cleaning/formatting.py

Corrección de formato de datos: columnas numéricas o de fecha que quedaron
guardadas como texto (típico al leer un Excel/CSV mal exportado), más
limpieza de nombres de columna y de texto libre.
Sin dependencia de LLM — pandas puro, testeable en aislado.

auto_correct_format se apoya en quality/profiler.py (profile_column) para
detectar automáticamente qué columnas están mal tipadas — el usuario no
necesita indicar cuáles.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Any

import pandas as pd

from quality.profiler import profile_column

_CASES_VALIDOS = ("lower", "upper", "title")


@dataclass
class FormatCorrectionResult:
    """Resumen de la corrección de formato aplicada."""
    columns_processed: list[str] = field(default_factory=list)
    columns_converted: dict[str, str] = field(default_factory=dict)  # columna -> 'numeric' | 'datetime'
    columns_unchanged: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serializa a dict plano — lo que tool_definitions.py le devuelve al LLM."""
        return asdict(self)


def auto_correct_format(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, FormatCorrectionResult]:
    """
    Detecta y corrige columnas con tipo de dato mal asignado (número o
    fecha guardados como texto), usando la misma heurística de
    quality/profiler.py para no duplicar lógica de inferencia de tipos.

    columns: columnas a revisar. Si es None, revisa todas.

    Las columnas cuyo tipo ya es correcto (o que son texto libre genuino)
    quedan registradas en columns_unchanged, sin tocarlas.
    """
    df_nuevo = df.copy()
    target_columns = columns if columns is not None else df_nuevo.columns.tolist()
    resultado = FormatCorrectionResult()

    for col in target_columns:
        if col not in df_nuevo.columns:
            continue

        resultado.columns_processed.append(col)
        perfil = profile_column(df_nuevo[col])

        if perfil.inferred_type == "numeric_as_text":
            df_nuevo[col] = pd.to_numeric(df_nuevo[col], errors="coerce")
            resultado.columns_converted[col] = "numeric"
        elif perfil.inferred_type == "datetime_as_text":
            df_nuevo[col] = pd.to_datetime(df_nuevo[col], errors="coerce", format="mixed")
            resultado.columns_converted[col] = "datetime"
        else:
            resultado.columns_unchanged.append(col)

    return df_nuevo, resultado


def _a_snake_case(texto: str) -> str:
    """Normaliza un texto a snake_case: sin tildes, sin espacios, minúsculas."""
    sin_tildes = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    limpio = re.sub(r"[^0-9a-zA-Z]+", "_", sin_tildes.strip())
    limpio = re.sub(r"_+", "_", limpio).strip("_").lower()
    return limpio or "columna_sin_nombre"


def clean_column_names(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Normaliza los nombres de columna a snake_case (saca tildes, espacios
    extra, mayúsculas mezcladas).

    Devuelve (df_nuevo, cambios), donde cambios es un dict
    {nombre_original: nombre_nuevo} solo con las columnas que sí cambiaron.
    """
    df_nuevo = df.copy()
    cambios: dict[str, str] = {}
    nuevas_columnas = []

    for col in df_nuevo.columns:
        nombre_original = str(col)
        nombre_nuevo = _a_snake_case(nombre_original)
        nuevas_columnas.append(nombre_nuevo)
        if nombre_nuevo != nombre_original:
            cambios[nombre_original] = nombre_nuevo

    df_nuevo.columns = nuevas_columnas
    return df_nuevo, cambios


def clean_text_column(
    df: pd.DataFrame,
    column: str,
    case: str | None = None,
) -> pd.DataFrame:
    """
    Recorta espacios y opcionalmente normaliza mayúsculas/minúsculas en
    una columna de texto.

    case: 'lower', 'upper', 'title' o None (solo recorta espacios).

    Lanza ValueError si la columna no existe o si case no es válido.
    """
    if column not in df.columns:
        raise ValueError(f"Columna no encontrada: {column!r}")
    if case is not None and case not in _CASES_VALIDOS:
        raise ValueError(f"case inválido: {case!r}. Usar uno de {_CASES_VALIDOS} o None.")

    df_nuevo = df.copy()
    serie = df_nuevo[column].astype(str).str.strip()

    if case == "lower":
        serie = serie.str.lower()
    elif case == "upper":
        serie = serie.str.upper()
    elif case == "title":
        serie = serie.str.title()

    df_nuevo[column] = serie
    return df_nuevo


if __name__ == "__main__":
    # Ejemplo rápido para probar el módulo de forma aislada.
    df_ejemplo = pd.DataFrame({
        " Nombre Cliente ": ["  ana  ", "LUIS", "marta "],
        "Monto ($)": ["100.5", "200", "50.25"],
        "Fecha Registro": ["2024-01-15", "2024-01-16", "2024-02-01"],
    })

    print("--- auto_correct_format ---")
    df_res, resumen = auto_correct_format(df_ejemplo)
    print(df_res.dtypes)
    print(resumen.to_dict())

    print("\n--- clean_column_names ---")
    df_res2, cambios = clean_column_names(df_res)
    print(df_res2.columns.tolist())
    print(cambios)

    print("\n--- clean_text_column (case='title') ---")
    df_res3 = clean_text_column(df_res2, column="nombre_cliente", case="title")
    print(df_res3)