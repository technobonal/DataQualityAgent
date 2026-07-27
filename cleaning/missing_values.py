"""
cleaning/missing_values.py

Identificación y tratamiento de valores faltantes.
Sin dependencia de LLM — pandas puro, testeable en aislado.

Además de los NaN "de verdad" que pandas reconoce, detecta marcadores de
texto que en la práctica también representan un dato faltante (ej. una
celda de Excel con "?", "N/A" o "-" en vez de estar realmente vacía).

MEJORA: identify_missing() ahora acepta min_null_pct, para que el LLM
pueda pedir directo "columnas con >=60% de faltantes" y reciba SOLO esas
columnas, en vez de la lista completa del dataset. Antes, con datasets
de muchas columnas, el recorte genérico de core/agent.py
(_recortar_para_llm, que corta listas largas a los primeros 10
elementos para no gastar tokens de más) truncaba por ORDEN DE
APARICIÓN -- si las columnas con muchos faltantes estaban después de la
posición 10, el LLM nunca las veía y terminaba sin poder responder.
Además, el resultado ahora viene ORDENADO de mayor a menor % de
faltantes por defecto: así, aunque el LLM pida el reporte SIN filtro
(ej. "dame un resumen de nulos"), lo que sobrevive a ese mismo recorte
genérico son las columnas más problemáticas, no las primeras 10 al
azar.
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
    values_filled: dict[str, int] = field(default_factory=dict)  # columna -> CANTIDAD de nulos rellenados
    fill_values: dict[str, Any] = field(default_factory=dict)  # columna -> VALOR real usado para rellenar

    def to_dict(self) -> dict[str, Any]:
        """Serializa a dict plano — lo que tool_definitions.py le devuelve al LLM."""
        return asdict(self)


def identify_missing(
    df: pd.DataFrame,
    min_null_pct: float | None = None,
) -> list[MissingReport]:
    """
    Genera un reporte de valores faltantes por columna: nulos reales
    (NaN) + marcadores de texto que actúan como nulo disfrazado.

    min_null_pct: si se pasa (0-100), devuelve SOLO las columnas cuyo
                  null_pct sea mayor o igual a ese umbral -- para pedidos
                  puntuales tipo "columnas con al menos 60% de faltantes".
                  Considera tanto nulos reales como marcadores de texto
                  como parte del % (null_pct + marker_pct combinado),
                  ya que ambos representan un dato faltante en la práctica.

    El resultado SIEMPRE viene ordenado de mayor a menor % de faltantes
    (nulos + marcadores combinados), para que las columnas más
    problemáticas queden primero -- relevante porque el LLM solo ve un
    máximo de elementos por lista (ver _recortar_para_llm en
    core/agent.py).
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

    # % combinado (nulos reales + marcadores de texto) -- lo que el
    # usuario percibe como "faltante" en la práctica, aunque
    # null_pct por sí solo no lo refleje si el dato está "disfrazado"
    # como "?", "N/A", etc.
    def _pct_combinado(r: MissingReport) -> float:
        if total == 0:
            return 0.0
        return round(((r.null_count + r.marker_count) / total) * 100, 2)

    reportes.sort(key=_pct_combinado, reverse=True)

    if min_null_pct is not None:
        reportes = [r for r in reportes if _pct_combinado(r) >= min_null_pct]

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
            resultado.fill_values[col] = round(float(valor), 4)
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
            valor = moda.iloc[0]
            df_nuevo[col] = df_nuevo[col].fillna(valor)
            resultado.values_filled[col] = n_nulos
            resultado.fill_values[col] = valor if not isinstance(valor, (int, float)) else round(float(valor), 4)
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
            resultado.fill_values[col] = constant_value
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
    # Ejemplo con muchas columnas, simulando el caso real: solo un par
    # superan el 60% de faltantes, y NO están al principio del dataset
    # -- justo el escenario que el recorte por orden de aparición no
    # podía resolver antes de este fix.
    import random
    random.seed(0)

    columnas = {"id": list(range(1, 21))}
    for i in range(1, 16):
        # la mayoría con pocos o cero nulos
        columnas[f"col_{i}"] = [random.choice([1, 2, 3, None]) if i in (11, 14) else i for i in range(20)]
    # dos columnas al FINAL con muchísimos faltantes
    columnas["col_muy_incompleta_1"] = [None] * 15 + [1, 2, 3, 4, 5]  # 75% nulos
    columnas["col_muy_incompleta_2"] = [None] * 13 + [1, 2, 3, 4, 5, 6, 7]  # 65% nulos

    df_ejemplo = pd.DataFrame(columnas)

    print("--- identify_missing SIN filtro (primeras 5 del resultado ordenado) ---")
    for r in identify_missing(df_ejemplo)[:5]:
        print(f"  {r.column:24s} null_pct={r.null_pct}%")

    print("\n--- identify_missing CON min_null_pct=60 ---")
    for r in identify_missing(df_ejemplo, min_null_pct=60):
        print(f"  {r.column:24s} null_pct={r.null_pct}%")