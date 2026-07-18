"""
tools/tool_definitions.py

Schemas de tool calling (formato OpenAI-style, compatible con Groq vía
LiteLLM) + dispatcher. Cada feature de limpieza/calidad se expone acá
como una función que el LLM puede invocar según lo que pida el usuario.

Patrón para sumar una feature nueva:
  1. Implementar la lógica en cleaning/<feature>.py (sin LLM, testeable sola)
  2. Agregar su schema a TOOLS_SCHEMA (describe cuándo debe usarla el LLM)
  3. Agregar su rama en dispatch_tool (ejecuta y devuelve JSON-serializable)
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from cleaning.missing_values import identify_missing, handle_missing_values
from cleaning.formatting import auto_correct_format, clean_column_names, clean_text_column
from cleaning.duplicates import identify_duplicates, handle_duplicates
from cleaning.standardization import standardize_columns
from cleaning.normalization import normalize_columns
from cleaning.binning import bin_column
from cleaning.indicators import create_indicator_variables
from quality.outliers import detect_outliers_iqr, handle_outliers
from quality.report import generate_quality_report
from quality.recommendations import recommend_actions


TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "identificar_valores_faltantes",
            "description": (
                "Identifica y cuenta los valores faltantes por columna en el dataset "
                "cargado por el usuario. Usar cuando pregunte cosas como "
                "'¿qué columnas tienen datos faltantes?' o '¿cuántos nulos hay?'."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tratar_valores_faltantes",
            "description": (
                "Aplica una estrategia para tratar los valores faltantes del dataset "
                "(rellenar o eliminar). Usar cuando el usuario pida 'rellenar los nulos', "
                "'eliminar filas con datos faltantes', 'completar con el promedio', etc. "
                "No usar para corregir formato o normalizar — eso son otras tools."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "enum": [
                            "drop_rows", "drop_columns", "mean", "median",
                            "mode", "constant", "ffill", "bfill",
                        ],
                        "description": "Estrategia de tratamiento a aplicar.",
                    },
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columnas a tratar. Si se omite, aplica a todas.",
                    },
                    "constant_value": {
                        "description": "Valor a usar cuando strategy='constant'.",
                    },
                },
                "required": ["strategy"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "corregir_formato_datos",
            "description": (
                "Detecta y corrige columnas con el tipo de dato mal asignado: números "
                "o fechas que quedaron guardados como texto. Usar cuando el usuario pida "
                "'corregir el formato de mis datos', 'esta columna debería ser numérica', etc. "
                "No requiere que el usuario indique qué columnas están mal — se detectan solas."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columnas a revisar. Si se omite, revisa todas.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "limpiar_nombres_columnas",
            "description": (
                "Normaliza los nombres de columna del dataset (saca espacios extra, "
                "tildes, mayúsculas mezcladas) a snake_case. Usar cuando el usuario pida "
                "'limpiar los nombres de las columnas' o note nombres desprolijos."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "limpiar_columna_texto",
            "description": (
                "Recorta espacios y normaliza mayúsculas/minúsculas en una columna de "
                "texto específica. Usar cuando el usuario pida 'sacar los espacios de esta "
                "columna' o 'poner todo en mayúsculas/minúsculas'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Nombre de la columna a limpiar."},
                    "case": {
                        "type": "string",
                        "enum": ["lower", "upper", "title"],
                        "description": "Formato de mayúsculas deseado. Si se omite, solo recorta espacios.",
                    },
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estandarizar_columnas",
            "description": (
                "Aplica estandarización z-score (media=0, desvío=1) a columnas numéricas. "
                "Usar cuando el usuario pida 'estandarizar', 'poner en la misma escala usando "
                "z-score' o mencione preparar datos para un modelo sensible a la escala "
                "(regresión, KNN, SVM, redes neuronales). No usar para llevar valores a un "
                "rango fijo como [0,1] — para eso está 'normalizar_columnas'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columnas numéricas a estandarizar. Si se omite, aplica a todas las numéricas.",
                    },
                    "suffix": {
                        "type": "string",
                        "description": (
                            "Sufijo para crear columnas nuevas (ej. '_z') en vez de "
                            "sobrescribir las originales. Si se omite, sobrescribe in place."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "normalizar_columnas",
            "description": (
                "Aplica normalización min-max a columnas numéricas, reescalando a un rango "
                "fijo (por defecto [0,1]). Usar cuando el usuario pida 'normalizar', 'llevar "
                "los valores entre 0 y 1' o similar. No usar para z-score — para eso está "
                "'estandarizar_columnas'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columnas numéricas a normalizar. Si se omite, aplica a todas las numéricas.",
                    },
                    "range_min": {
                        "type": "number",
                        "description": "Límite inferior del rango destino. Por defecto 0.",
                    },
                    "range_max": {
                        "type": "number",
                        "description": "Límite superior del rango destino. Por defecto 1.",
                    },
                    "suffix": {
                        "type": "string",
                        "description": (
                            "Sufijo para crear columnas nuevas (ej. '_norm') en vez de "
                            "sobrescribir las originales. Si se omite, sobrescribe in place."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "discretizar_columna",
            "description": (
                "Discretiza (binning) una columna numérica en rangos o categorías. Usar "
                "cuando el usuario pida 'agrupar edades en rangos', 'convertir esta columna "
                "en categorías', 'discretizar', o pida bins de igual ancho o igual frecuencia. "
                "Siempre crea una columna nueva, nunca sobrescribe la original."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Columna numérica a discretizar.",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["equal_width", "equal_frequency", "custom"],
                        "description": (
                            "'equal_width': rangos de igual ancho. "
                            "'equal_frequency': misma cantidad de filas por bin. "
                            "'custom': el usuario define los límites exactos en 'bins'."
                        ),
                    },
                    "bins": {
                        "description": (
                            "Cantidad de bins (entero) para equal_width/equal_frequency, "
                            "o lista de límites (ej. [0, 30, 60, 100]) para method='custom'."
                        ),
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Nombres para cada bin (ej. ['joven','adulto','mayor']). Opcional.",
                    },
                    "new_column": {
                        "type": "string",
                        "description": "Nombre de la columna resultado. Si se omite, usa '{column}_bin'.",
                    },
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crear_variables_indicadoras",
            "description": (
                "Convierte columnas categóricas/texto en variables indicadoras (dummy / "
                "one-hot encoding): una columna binaria (0/1) por cada categoría. Usar "
                "cuando el usuario pida 'convertir esta columna categórica en números', "
                "'one-hot encoding', 'dummies' o prepare datos categóricos para un modelo. "
                "Si la columna tiene muchas categorías distintas (alta cardinalidad), "
                "considerar pasar top_n para agrupar las menos frecuentes en 'otros'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columnas categóricas a codificar. Si se omite, aplica a todas las de tipo texto/categórico.",
                    },
                    "drop_first": {
                        "type": "boolean",
                        "description": (
                            "Si True, descarta la primera categoría de cada columna para "
                            "evitar colinealidad perfecta (recomendado para regresión lineal). "
                            "Por defecto False."
                        ),
                    },
                    "top_n": {
                        "type": "integer",
                        "description": (
                            "Si se pasa, conserva solo las top_n categorías más frecuentes "
                            "y agrupa el resto bajo 'otros'. Útil para columnas con muchas "
                            "categorías distintas."
                        ),
                    },
                    "drop_original": {
                        "type": "boolean",
                        "description": "Si True (default), elimina la columna categórica original tras codificarla.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "identificar_outliers",
            "description": (
                "Detecta valores atípicos (outliers) en columnas numéricas usando el "
                "método IQR. Usar cuando el usuario pregunte '¿hay valores raros/atípicos "
                "en mis datos?', '¿qué columnas tienen outliers?', etc. No modifica el "
                "dataset — solo informa. Para corregirlos usar 'tratar_outliers'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columnas numéricas a revisar. Si se omite, revisa todas las numéricas.",
                    },
                    "multiplier": {
                        "type": "number",
                        "description": "Qué tan estricto es el criterio IQR. 1.5 (default) es el estándar; usar 3.0 para outliers extremos.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tratar_outliers",
            "description": (
                "Corrige los outliers detectados por el método IQR. Usar cuando el "
                "usuario pida 'corregir los valores atípicos', 'recortar los outliers' "
                "o 'eliminar las filas con valores raros'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "enum": ["cap", "remove"],
                        "description": (
                            "'cap': recorta (winsoriza) los valores fuera de rango, sin perder filas. "
                            "'remove': elimina las filas con outliers."
                        ),
                    },
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columnas numéricas a tratar. Si se omite, aplica a todas las numéricas.",
                    },
                    "multiplier": {
                        "type": "number",
                        "description": "Qué tan estricto es el criterio IQR. 1.5 (default) es el estándar.",
                    },
                },
                "required": ["strategy"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generar_reporte_calidad",
            "description": (
                "Genera un reporte de calidad completo del dataset: nulos, duplicados, "
                "tipos de dato mal asignados, outliers, y un puntaje de calidad de 0 a "
                "100. Usar cuando el usuario pida '¿cómo está la calidad de mis datos?', "
                "'dame un resumen general', 'analizá el dataset completo', etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "duplicate_subset": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columnas específicas para chequear duplicados. Si se omite, compara la fila completa.",
                    },
                    "outlier_columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columnas específicas para chequear outliers. Si se omite, revisa todas las numéricas.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "diagnosticar_y_recomendar",
            "description": (
                "Genera el reporte de calidad completo Y, a partir de él, la lista de "
                "acciones que este archivo en particular necesita (ordenadas según el "
                "flujo clásico de limpieza) más las tools que NO hace falta usar para "
                "este dataset. Es la tool ideal para el primer diagnóstico apenas el "
                "usuario sube un archivo — reemplaza tener que llamar "
                "'generar_reporte_calidad' y decidir las acciones por separado."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "duplicate_subset": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columnas específicas para chequear duplicados. Si se omite, compara la fila completa.",
                    },
                    "outlier_columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columnas específicas para chequear outliers. Si se omite, revisa todas las numéricas.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "identificar_duplicados",
            "description": (
                "Detecta filas duplicadas en el dataset, sin modificarlo. Usar cuando "
                "el usuario pregunte '¿tengo filas repetidas?' o '¿cuántos duplicados hay?'. "
                "Para eliminarlos usar 'tratar_duplicados'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subset": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Columnas específicas para considerar al comparar filas "
                            "(ej. solo ['id', 'email']). Si se omite, compara la fila completa."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tratar_duplicados",
            "description": (
                "Elimina filas duplicadas del dataset. Usar cuando el usuario pida "
                "'eliminar duplicados', 'sacar las filas repetidas', 'quedarme con un "
                "solo registro por id', etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "enum": ["keep_first", "keep_last", "remove_all"],
                        "description": (
                            "'keep_first': conserva la primera aparición de cada duplicado. "
                            "'keep_last': conserva la última. "
                            "'remove_all': elimina TODAS las copias, sin conservar ninguna "
                            "(útil cuando el duplicado en sí indica un dato sucio)."
                        ),
                    },
                    "subset": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columnas específicas para considerar. Si se omite, compara la fila completa.",
                    },
                },
                "required": ["strategy"],
            },
        },
    },
]


def dispatch_tool(name: str, arguments: dict[str, Any], df: pd.DataFrame) -> tuple[Any, pd.DataFrame | None]:
    """
    Router central: recibe el nombre de la tool + argumentos del LLM,
    ejecuta la función correspondiente y devuelve:
        (resultado_json_serializable, df_actualizado_o_None)

    df_actualizado va a ser distinto de None solo si la tool modificó el
    dataset (ej. tratar_valores_faltantes) — ahí es responsabilidad del
    caller (core/agent.py) actualizar el DataFrame en el estado de sesión.
    """
    if name == "identificar_valores_faltantes":
        reportes = identify_missing(df)
        return [asdict(r) for r in reportes], None

    if name == "tratar_valores_faltantes":
        df_nuevo, resultado = handle_missing_values(
            df,
            strategy=arguments["strategy"],
            columns=arguments.get("columns"),
            constant_value=arguments.get("constant_value"),
        )
        return resultado.to_dict(), df_nuevo

    if name == "corregir_formato_datos":
        df_nuevo, resultado = auto_correct_format(df, columns=arguments.get("columns"))
        return resultado.to_dict(), df_nuevo

    if name == "limpiar_nombres_columnas":
        df_nuevo, cambios = clean_column_names(df)
        return {"columnas_renombradas": cambios}, df_nuevo

    if name == "limpiar_columna_texto":
        df_nuevo = clean_text_column(df, column=arguments["column"], case=arguments.get("case"))
        return {"columna": arguments["column"], "case": arguments.get("case", None)}, df_nuevo

    if name == "identificar_duplicados":
        reporte = identify_duplicates(df, subset=arguments.get("subset"))
        return reporte.to_dict(), None

    if name == "tratar_duplicados":
        df_nuevo, resultado = handle_duplicates(
            df,
            strategy=arguments["strategy"],
            subset=arguments.get("subset"),
        )
        return resultado.to_dict(), df_nuevo

    if name == "estandarizar_columnas":
        df_nuevo, resultado = standardize_columns(
            df,
            columns=arguments.get("columns"),
            suffix=arguments.get("suffix"),
        )
        return resultado.to_dict(), df_nuevo

    if name == "normalizar_columnas":
        df_nuevo, resultado = normalize_columns(
            df,
            columns=arguments.get("columns"),
            range_min=arguments.get("range_min", 0.0),
            range_max=arguments.get("range_max", 1.0),
            suffix=arguments.get("suffix"),
        )
        return resultado.to_dict(), df_nuevo

    if name == "discretizar_columna":
        df_nuevo, resultado = bin_column(
            df,
            column=arguments["column"],
            method=arguments.get("method", "equal_width"),
            bins=arguments.get("bins", 5),
            labels=arguments.get("labels"),
            new_column=arguments.get("new_column"),
        )
        return resultado.to_dict(), df_nuevo

    if name == "crear_variables_indicadoras":
        df_nuevo, resultado = create_indicator_variables(
            df,
            columns=arguments.get("columns"),
            drop_first=arguments.get("drop_first", False),
            top_n=arguments.get("top_n"),
            drop_original=arguments.get("drop_original", True),
        )
        return resultado.to_dict(), df_nuevo

    if name == "identificar_outliers":
        reportes = detect_outliers_iqr(
            df,
            columns=arguments.get("columns"),
            multiplier=arguments.get("multiplier", 1.5),
        )
        return [asdict(r) for r in reportes], None

    if name == "tratar_outliers":
        df_nuevo, resultado = handle_outliers(
            df,
            strategy=arguments["strategy"],
            columns=arguments.get("columns"),
            multiplier=arguments.get("multiplier", 1.5),
        )
        return resultado.to_dict(), df_nuevo

    if name == "generar_reporte_calidad":
        reporte = generate_quality_report(
            df,
            duplicate_subset=arguments.get("duplicate_subset"),
            outlier_columns=arguments.get("outlier_columns"),
        )
        return reporte.to_dict(), None

    if name == "diagnosticar_y_recomendar":
        reporte = generate_quality_report(
            df,
            duplicate_subset=arguments.get("duplicate_subset"),
            outlier_columns=arguments.get("outlier_columns"),
        )
        recomendacion = recommend_actions(reporte.to_dict())
        return {
            "diagnostico": reporte.to_dict(),
            "recomendacion": recomendacion.to_dict(),
        }, None

    raise ValueError(f"Tool desconocida: {name!r}")