"""
quality/recommendations.py

Motor de recomendación: traduce el diagnóstico de quality/report.py en una
lista de acciones concretas — cada una apuntando al nombre exacto de la
tool en tools/tool_definitions.py que hay que invocar.
Sin dependencia de LLM — pandas puro, testeable en aislado.

Es el puente entre "qué está mal en este archivo" (report.py) y
"qué tool específica lo arregla" (tool_definitions.py). El objetivo es
que el diagnóstico se adapte a cada dataset: si un archivo no tiene
columnas categóricas, por ejemplo, "crear_variables_indicadoras" no se
sugiere — queda en not_needed con la razón.

Orden de las acciones necesarias: sigue el flujo clásico de data
wrangling (corregir tipos -> revisar duplicados -> tratar nulos ->
tratar outliers -> escalar -> codificar categóricas), no el orden en
que aparecen en TOOLS_SCHEMA.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

# Umbral a partir del cual un problema se considera de prioridad "alta"
# en vez de "media" (ej. más de este % de nulos/duplicados en promedio).
_UMBRAL_PRIORIDAD_ALTA_PCT = 20.0
_UMBRAL_DUPLICADOS_ALTA_PCT = 5.0

_ORDEN_PRIORIDAD = {"alta": 0, "media": 1, "baja": 2}


@dataclass
class RecommendedAction:
    """Una acción que el diagnóstico determinó que SÍ hace falta."""
    tool_name: str | None  # None si el problema no tiene todavía una tool automática (ej. duplicados)
    reason: str
    priority: str  # 'alta', 'media', 'baja'
    affected_columns: list[str] = field(default_factory=list)


@dataclass
class SkippedAction:
    """Una tool/acción que el diagnóstico determinó que NO hace falta para este archivo."""
    tool_name: str
    reason: str


@dataclass
class RecommendationResult:
    """Resultado completo del motor de recomendación."""
    needed_actions: list[RecommendedAction] = field(default_factory=list)
    not_needed: list[SkippedAction] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serializa a dict plano — lo que Streamlit/core/agent.py consumen."""
        return asdict(self)


def recommend_actions(quality_report: dict[str, Any]) -> RecommendationResult:
    """
    Analiza un reporte de calidad (el dict que devuelve
    quality.report.generate_quality_report(df).to_dict()) y arma la lista
    de acciones necesarias, en el orden clásico de data wrangling, más la
    lista de tools que este archivo en particular no necesita.
    """
    profile = quality_report["profile"]
    columnas = profile["columns"]
    duplicados = profile["duplicates"]
    outliers = quality_report["outliers"]

    resultado = RecommendationResult()

    # --- 1. Formato de datos (numérico/fecha guardados como texto) ---
    mal_tipadas = [
        nombre for nombre, c in columnas.items()
        if c["inferred_type"] in ("numeric_as_text", "datetime_as_text")
    ]
    if mal_tipadas:
        resultado.needed_actions.append(RecommendedAction(
            tool_name="corregir_formato_datos",
            reason=f"{len(mal_tipadas)} columna(s) tienen números o fechas guardados como texto.",
            priority="alta",
            affected_columns=mal_tipadas,
        ))
    else:
        resultado.not_needed.append(SkippedAction(
            tool_name="corregir_formato_datos",
            reason="Todas las columnas ya tienen el tipo de dato correcto.",
        ))

    # --- 2. Duplicados ---
    if duplicados["total_duplicates"] > 0:
        prioridad = "alta" if duplicados["duplicate_pct"] > _UMBRAL_DUPLICADOS_ALTA_PCT else "media"
        resultado.needed_actions.append(RecommendedAction(
            tool_name="tratar_duplicados",
            reason=(
                f"Se detectaron {duplicados['total_duplicates']} filas duplicadas "
                f"({duplicados['duplicate_pct']}%)."
            ),
            priority=prioridad,
        ))
    else:
        resultado.not_needed.append(SkippedAction(
            tool_name="tratar_duplicados",
            reason="No se encontraron filas duplicadas.",
        ))

    # --- 3. Valores faltantes ---
    columnas_con_nulos = [nombre for nombre, c in columnas.items() if c["null_count"] > 0]
    if columnas_con_nulos:
        pct_promedio = sum(columnas[c]["null_pct"] for c in columnas_con_nulos) / len(columnas_con_nulos)
        prioridad = "alta" if pct_promedio > _UMBRAL_PRIORIDAD_ALTA_PCT else "media"
        resultado.needed_actions.append(RecommendedAction(
            tool_name="tratar_valores_faltantes",
            reason=f"{len(columnas_con_nulos)} columna(s) tienen valores faltantes (promedio {pct_promedio:.1f}%).",
            priority=prioridad,
            affected_columns=columnas_con_nulos,
        ))
    else:
        resultado.not_needed.append(SkippedAction(
            tool_name="tratar_valores_faltantes",
            reason="No se encontraron valores faltantes.",
        ))

    # --- 4. Outliers ---
    columnas_con_outliers = [o["column"] for o in outliers if o["outlier_count"] > 0]
    if columnas_con_outliers:
        resultado.needed_actions.append(RecommendedAction(
            tool_name="tratar_outliers",
            reason=f"{len(columnas_con_outliers)} columna(s) numérica(s) tienen valores atípicos.",
            priority="media",
            affected_columns=columnas_con_outliers,
        ))
    else:
        resultado.not_needed.append(SkippedAction(
            tool_name="tratar_outliers",
            reason="No se detectaron valores atípicos relevantes.",
        ))

    # --- 5. Escalado (estandarizar/normalizar) ---
    # No es un "problema" de calidad en sí — es una preparación opcional
    # para modelado. Se sugiere solo si hay suficientes columnas numéricas
    # como para que escalar tenga sentido.
    numericas = [nombre for nombre, c in columnas.items() if c["inferred_type"] == "numeric"]
    if len(numericas) >= 2:
        resultado.needed_actions.append(RecommendedAction(
            tool_name="estandarizar_columnas",
            reason=(
                f"Hay {len(numericas)} columnas numéricas — si vas a usar estos datos "
                f"en un modelo (regresión, KNN, redes neuronales, etc.), conviene "
                f"escalarlas (estandarizar_columnas o normalizar_columnas)."
            ),
            priority="baja",
            affected_columns=numericas,
        ))
    else:
        resultado.not_needed.append(SkippedAction(
            tool_name="estandarizar_columnas / normalizar_columnas",
            reason="No hay suficientes columnas numéricas para que el escalado tenga impacto real.",
        ))

    # --- 6. Binning ---
    # Siempre es una transformación opcional según el análisis que se
    # quiera hacer, nunca "necesaria" por un problema de calidad detectado.
    resultado.not_needed.append(SkippedAction(
        tool_name="discretizar_columna",
        reason="Es una transformación opcional según el análisis que quieras hacer, no depende de un problema de calidad detectado.",
    ))

    # --- 7. Variables indicadoras (one-hot / dummies) ---
    categoricas = [nombre for nombre, c in columnas.items() if c["inferred_type"] == "categorical"]
    if categoricas:
        resultado.needed_actions.append(RecommendedAction(
            tool_name="crear_variables_indicadoras",
            reason=f"{len(categoricas)} columna(s) categórica(s) detectadas — conviene codificarlas si vas a usar los datos en un modelo.",
            priority="media",
            affected_columns=categoricas,
        ))
    else:
        resultado.not_needed.append(SkippedAction(
            tool_name="crear_variables_indicadoras",
            reason="No se detectaron columnas categóricas que requieran codificación.",
        ))

    # Orden final: prioridad alta -> media -> baja (respeta el flujo de
    # data wrangling dentro de cada nivel, porque ya las fuimos agregando
    # en ese orden).
    resultado.needed_actions.sort(key=lambda a: _ORDEN_PRIORIDAD.get(a.priority, 3))

    return resultado


if __name__ == "__main__":
    # Ejemplo rápido: un dataset con problemas variados + uno "limpio"
    # para confirmar que el motor se adapta a cada caso.
    import pandas as pd
    from quality.report import generate_quality_report

    print("=== Caso 1: dataset con varios problemas ===")
    df_sucio = pd.DataFrame({
        "id": [1, 2, 2, 3, None, 6],
        "nombre": ["Ana", "Luis", "Luis", "Marta", "Pedro", "Sofia"],
        "fecha_registro": ["2024-01-15", "2024-01-16", "2024-01-16", "no_es_fecha", "2024-02-01", "2024-03-10"],
        "monto": [100.5, 200.0, 200.0, None, 50.25, 999999.0],
        "ciudad": ["SP", "RJ", "RJ", "BH", "SP", "SP"],
    })
    reporte1 = generate_quality_report(df_sucio)
    recomendacion1 = recommend_actions(reporte1.to_dict())

    print("Acciones necesarias (en orden):")
    for a in recomendacion1.needed_actions:
        print(f"  [{a.priority.upper()}] {a.tool_name or 'SIN TOOL AUTOMÁTICA'} -> {a.reason}")
    print("\nNo hace falta:")
    for s in recomendacion1.not_needed:
        print(f"  {s.tool_name} -> {s.reason}")

    print("\n=== Caso 2: dataset ya limpio (para confirmar que no sobre-sugiere) ===")
    df_limpio = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "valor": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    reporte2 = generate_quality_report(df_limpio)
    recomendacion2 = recommend_actions(reporte2.to_dict())

    print("Acciones necesarias:", len(recomendacion2.needed_actions))
    for a in recomendacion2.needed_actions:
        print(f"  [{a.priority.upper()}] {a.tool_name} -> {a.reason}")
    print("No hace falta:", len(recomendacion2.not_needed))