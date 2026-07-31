"""
cleaning/formatting.py

Corrección de formato de datos: columnas numéricas o de fecha que quedaron
guardadas como texto (típico al leer un Excel/CSV mal exportado), más
limpieza de nombres de columna y de texto libre.
Sin dependencia de LLM — pandas puro, testeable en aislado.

auto_correct_format se apoya en quality/profiler.py (profile_column) para
detectar automáticamente qué columnas están mal tipadas — el usuario no
necesita indicar cuáles.

MEJORA: FormatCorrectionResult ahora incluye conteos totales explícitos
(total_processed, total_converted, total_unchanged). A diferencia de
missing_values.py/outliers.py/duplicates.py, acá el dato más importante
(columns_converted: qué columna se convirtió a qué tipo) es un DICT, y
el recorte genérico de listas largas en core/agent.py
(_recortar_para_llm) solo trunca LISTS, no dicts -- así que
columns_converted nunca se pierde, sin importar cuántas columnas tenga.
Lo único que sí puede truncarse son columns_processed/columns_unchanged
(listas de nombres). Los conteos totales garantizan que, aunque esas
listas se trunquen, el LLM siempre tenga el número exacto para responder
"¿cuántas columnas se convirtieron/no cambiaron?" sin tener que contar
elementos de una lista que podría estar incompleta.
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
    total_processed: int = 0
    total_converted: int = 0
    total_unchanged: int = 0

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

    resultado.total_processed = len(resultado.columns_processed)
    resultado.total_converted = len(resultado.columns_converted)
    resultado.total_unchanged = len(resultado.columns_unchanged)

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
    # Ejemplo con muchas columnas (más de 10) para confirmar que
    # total_converted/total_unchanged quedan exactos aunque las listas
    # columns_processed/columns_unchanged se trunquen después, en el
    # recorte genérico de core/agent.py (que acá no corremos, pero los
    # totales ya vienen resueltos de antes).
    columnas = {}
    for i in range(1, 13):
        columnas[f"num_como_texto_{i}"] = ["1", "2", "3"]  # todas numeric_as_text
    columnas["texto_libre"] = ["hola", "chau", "che"]  # queda sin cambios

    df_ejemplo = pd.DataFrame(columnas)

    print("--- auto_correct_format con 13 columnas ---")
    df_res, resumen = auto_correct_format(df_ejemplo)
    print(f"total_processed={resumen.total_processed}  "
          f"total_converted={resumen.total_converted}  "
          f"total_unchanged={resumen.total_unchanged}")
    print(f"columns_converted (dict completo, NUNCA se trunca): {len(resumen.columns_converted)} entradas")

    print("\n--- clean_column_names ---")
    df_ejemplo2 = pd.DataFrame({
        " Nombre Cliente ": ["  ana  ", "LUIS", "marta "],
        "Monto ($)": ["100.5", "200", "50.25"],
    })
    df_res2, cambios = clean_column_names(df_ejemplo2)
    print(df_res2.columns.tolist())
    print(cambios)