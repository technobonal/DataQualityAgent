"""
quality/report.py

Reporte de calidad consolidado: combina quality/profiler.py (nulos,
duplicados, tipos) y quality/outliers.py (valores atípicos) en un único
resultado, más un puntaje de calidad (0-100) y un resumen de problemas
en lenguaje simple.
Sin dependencia de LLM — pandas puro, testeable en aislado.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import pandas as pd

from quality.profiler import profile_dataframe
from quality.outliers import detect_outliers_iqr

# Peso de cada factor en el descuento de puntaje (sobre una base de 100).
_PESO_NULOS = 0.5
_PESO_DUPLICADOS = 0.5
_PESO_OUTLIERS = 0.3
_PENALIZACION_POR_COLUMNA_MAL_TIPADA = 2.0

# A partir de qué porcentaje un problema se considera "relevante" para el resumen.
_UMBRAL_AVISO_PCT = 5.0


@dataclass
class QualityReport:
    """Reporte de calidad consolidado de un DataFrame."""
    profile: dict[str, Any]
    outliers: list[dict[str, Any]]
    quality_score: float
    issues_summary: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serializa a dict plano — lo que tool_definitions.py le devuelve al LLM."""
        return asdict(self)


def _calcular_score_y_resumen(
    perfil,
    outlier_reports,
) -> tuple[float, list[str]]:
    """
    Calcula un puntaje de calidad (0-100) y arma un resumen de problemas
    en lenguaje simple, a partir del perfil y los outliers ya detectados.

    La lógica es intencionalmente simple y transparente (resta puntos por
    cada factor) para que el resultado sea fácil de explicar al usuario,
    en vez de un score "caja negra".
    """
    score = 100.0
    issues: list[str] = []

    # --- Nulos ---
    null_pcts = [c.null_pct for c in perfil.columns.values()]
    avg_null_pct = sum(null_pcts) / len(null_pcts) if null_pcts else 0.0
    score -= avg_null_pct * _PESO_NULOS
    if avg_null_pct > _UMBRAL_AVISO_PCT:
        issues.append(f"Hay en promedio {avg_null_pct:.1f}% de valores faltantes por columna.")

    # --- Duplicados ---
    score -= perfil.duplicates.duplicate_pct * _PESO_DUPLICADOS
    if perfil.duplicates.total_duplicates > 0:
        issues.append(
            f"Se detectaron {perfil.duplicates.total_duplicates} filas duplicadas "
            f"({perfil.duplicates.duplicate_pct}% del total)."
        )

    # --- Outliers ---
    outlier_pcts = [o.outlier_pct for o in outlier_reports]
    avg_outlier_pct = sum(outlier_pcts) / len(outlier_pcts) if outlier_pcts else 0.0
    score -= avg_outlier_pct * _PESO_OUTLIERS
    columnas_con_outliers = [o.column for o in outlier_reports if o.outlier_count > 0]
    if columnas_con_outliers:
        issues.append(
            f"Columnas con valores atípicos (outliers): {', '.join(columnas_con_outliers)}."
        )

    # --- Columnas mal tipadas (detectadas por profiler: numérico/fecha guardados como texto) ---
    columnas_mal_tipadas = [
        c.name for c in perfil.columns.values()
        if c.inferred_type in ("numeric_as_text", "datetime_as_text")
    ]
    if columnas_mal_tipadas:
        score -= len(columnas_mal_tipadas) * _PENALIZACION_POR_COLUMNA_MAL_TIPADA
        issues.append(
            f"Columnas con tipo de dato mal asignado: {', '.join(columnas_mal_tipadas)}."
        )

    if not issues:
        issues.append("No se detectaron problemas relevantes de calidad.")

    score = max(0.0, min(100.0, round(score, 2)))
    return score, issues


def generate_quality_report(
    df: pd.DataFrame,
    duplicate_subset: list[str] | None = None,
    outlier_columns: list[str] | None = None,
    outlier_multiplier: float = 1.5,
) -> QualityReport:
    """
    Punto de entrada principal: genera el reporte de calidad consolidado.

    duplicate_subset:   columnas específicas para chequear duplicados
                        (ver quality/profiler.py). Si es None, compara la fila completa.
    outlier_columns:    columnas específicas para chequear outliers. Si es
                        None, revisa todas las numéricas.
    outlier_multiplier: qué tan estricto es el criterio IQR (ver quality/outliers.py).

    Uso típico desde tools/tool_definitions.py:
        reporte = generate_quality_report(df)
        return reporte.to_dict()  # JSON-serializable para el LLM
    """
    perfil = profile_dataframe(df, duplicate_subset=duplicate_subset)
    outlier_reports = detect_outliers_iqr(df, columns=outlier_columns, multiplier=outlier_multiplier)

    score, issues = _calcular_score_y_resumen(perfil, outlier_reports)

    return QualityReport(
        profile=perfil.to_dict(),
        outliers=[asdict(o) for o in outlier_reports],
        quality_score=score,
        issues_summary=issues,
    )


if __name__ == "__main__":
    # Ejemplo rápido para probar el módulo de forma aislada, simulando un
    # dataset con varios problemas típicos a la vez.
    df_ejemplo = pd.DataFrame({
        "id": [1, 2, 2, 3, None, 6],
        "nombre": ["Ana", "Luis", "Luis", "Marta", "Pedro", "Sofia"],
        "fecha_registro": ["2024-01-15", "2024-01-16", "2024-01-16", "no_es_fecha", "2024-02-01", "2024-03-10"],
        "monto": [100.5, 200.0, 200.0, None, 50.25, 999999.0],  # 999999 es un outlier evidente
        "cantidad_como_texto": ["10", "20", "20", "no_aplica", "30", "15"],
    })

    reporte = generate_quality_report(df_ejemplo)

    print(f"Quality score: {reporte.quality_score}/100\n")
    print("Resumen de problemas:")
    for issue in reporte.issues_summary:
        print(f"  - {issue}")

    print(f"\nFilas: {reporte.profile['row_count']} | Columnas: {reporte.profile['column_count']}")
    print(f"Outliers detectados: {[o for o in reporte.outliers if o['outlier_count'] > 0]}")