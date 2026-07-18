"""
cleaning/indicators.py

Variables indicadoras (dummy/one-hot encoding) para columnas categóricas:
convierte cada categoría en su propia columna binaria (0/1).
Sin dependencia de LLM — pandas puro, testeable en aislado.

Útil porque la mayoría de los modelos de ML no pueden trabajar
directamente con texto/categorías — necesitan números.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import pandas as pd

# Si una columna tiene más categorías que esto, avisamos en el resultado
# (no bloqueamos, pero el usuario debería saber que va a crear muchas columnas).
_HIGH_CARDINALITY_WARNING = 20


@dataclass
class IndicatorResult:
    """Resumen de las variables indicadoras generadas."""
    columns_processed: list[str] = field(default_factory=list)
    columns_skipped: list[str] = field(default_factory=list)  # no encontradas o ya numéricas sin categorías
    new_columns: list[str] = field(default_factory=list)
    categories_per_column: dict[str, list[str]] = field(default_factory=dict)
    high_cardinality_warnings: dict[str, int] = field(default_factory=dict)  # columna -> cantidad de categorías

    def to_dict(self) -> dict[str, Any]:
        """Serializa a dict plano — lo que tool_definitions.py le devuelve al LLM."""
        return asdict(self)


def create_indicator_variables(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    drop_first: bool = False,
    top_n: int | None = None,
    other_label: str = "otros",
    drop_original: bool = True,
) -> tuple[pd.DataFrame, IndicatorResult]:
    """
    Genera variables indicadoras (0/1) para columnas categóricas.

    columns:       columnas a codificar. Si es None, aplica a todas las
                    columnas de tipo texto/categórico (object, string, category).
    drop_first:    si True, descarta la primera categoría de cada columna para
                    evitar colinealidad perfecta (típico en regresión lineal).
    top_n:         si se pasa, conserva solo las top_n categorías más
                    frecuentes de cada columna y agrupa el resto bajo
                    other_label. Útil para columnas de alta cardinalidad
                    (ej. una columna "ciudad" con 200 valores distintos).
    other_label:   nombre de la categoría "resto" cuando se usa top_n.
    drop_original: si True (default), elimina la columna categórica original
                    después de generar sus indicadoras.

    Columnas no encontradas o que no son de tipo texto/categórico se
    saltean y quedan registradas en columns_skipped.
    """
    df_nuevo = df.copy()
    target_columns = (
        columns
        if columns is not None
        else df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    )

    resultado = IndicatorResult()

    for col in target_columns:
        if col not in df_nuevo.columns:
            resultado.columns_skipped.append(col)
            continue
        if not (
            pd.api.types.is_object_dtype(df_nuevo[col])
            or pd.api.types.is_string_dtype(df_nuevo[col])
            or isinstance(df_nuevo[col].dtype, pd.CategoricalDtype)
        ):
            resultado.columns_skipped.append(col)
            continue

        serie = df_nuevo[col]
        n_categorias = serie.nunique(dropna=True)

        if n_categorias > _HIGH_CARDINALITY_WARNING:
            resultado.high_cardinality_warnings[col] = int(n_categorias)

        # Alta cardinalidad: agrupar categorías poco frecuentes bajo "otros"
        # para no explotar en cientos de columnas nuevas.
        if top_n is not None and n_categorias > top_n:
            top_categorias = serie.value_counts().nlargest(top_n).index
            serie = serie.where(serie.isin(top_categorias), other=other_label)

        dummies = pd.get_dummies(serie, prefix=col, drop_first=drop_first, dtype=int)
        df_nuevo = pd.concat([df_nuevo, dummies], axis=1)

        if drop_original:
            df_nuevo = df_nuevo.drop(columns=[col])

        resultado.columns_processed.append(col)
        resultado.new_columns.extend(dummies.columns.tolist())
        resultado.categories_per_column[col] = dummies.columns.tolist()

    return df_nuevo, resultado


if __name__ == "__main__":
    # Ejemplo rápido para probar el módulo de forma aislada.
    df_ejemplo = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "ciudad": ["SP", "RJ", "SP", "BH", "SP"],
        "plan": ["basico", "premium", "premium", "basico", "gratis"],
    })

    print("--- básico (sin drop_first) ---")
    df_res, resumen = create_indicator_variables(df_ejemplo, columns=["ciudad", "plan"])
    print(df_res)
    print(resumen.to_dict())

    print("\n--- con drop_first=True (evita colinealidad) ---")
    df_res2, resumen2 = create_indicator_variables(df_ejemplo, columns=["plan"], drop_first=True)
    print(df_res2)
    print(resumen2.to_dict())

    print("\n--- con top_n (alta cardinalidad simulada) ---")
    df_ciudades = pd.DataFrame({
        "ciudad": ["SP", "RJ", "BH", "SP", "Curitiba", "Recife", "SP", "RJ", "Salvador", "SP"],
    })
    df_res3, resumen3 = create_indicator_variables(df_ciudades, columns=["ciudad"], top_n=3)
    print(df_res3)
    print(resumen3.to_dict())