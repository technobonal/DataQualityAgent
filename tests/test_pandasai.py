"""
tests/test_pandasai.py

Cubre analysis/pandasai_wrapper.py (las 4 funciones públicas de
generación + los 2 helpers de metadata) sin llamar a ningún LLM real:
se monkeypatchea chat_completion/extract_text dentro del propio módulo
pandasai_wrapper (no en core.llm_client -- ya están importados por
referencia ahí, parchear el módulo original no alcanza).

Estructura pedida en el plan:
  - Q/A por función: análisis simple, corrección de tipos (dtype
    cambia), eliminación de filas (len baja, nunca sube), gráfico
    (isinstance(output, Figure)).
  - Casos de rechazo por sandbox (sigue sin reintentar).
  - Caso de error real recuperable (reintenta 1 vez).
"""
from __future__ import annotations

import pandas as pd
import pytest
from matplotlib.figure import Figure

import analysis.pandasai_wrapper as w


@pytest.fixture
def df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "edad": ["22", "31", "47", "35", "29"],  # texto a propósito, para poder testear corrección de tipo
        "ciudad": ["SP", "RJ", "SP", None, "BH"],
    })


class RespuestaFake:
    def __init__(self, texto: str):
        self.texto = texto


def _mock_fijo(codigo: str):
    """Arma un mock de chat_completion que siempre devuelve el mismo código, sin importar el prompt."""
    def _mock(mensajes, rate_limiter=None, tools=None, temperature=0.0):
        return RespuestaFake(codigo)
    return _mock


@pytest.fixture(autouse=True)
def _parchear_extract_text(monkeypatch):
    # extract_text ya está importado por referencia en pandasai_wrapper --
    # parchear core.llm_client.extract_text NO alcanza, hay que parchear
    # el nombre dentro del propio módulo w.
    monkeypatch.setattr(w, "extract_text", lambda r: r.texto)


# ---------------------------------------------------------------------
# Q/A por función
# ---------------------------------------------------------------------

def test_analizar_simple(df, monkeypatch):
    monkeypatch.setattr(w, "chat_completion", _mock_fijo("resultado = df['edad'].astype(int).mean()"))
    r = w.analizar_con_pandasai(df, "¿cuál es el promedio de edad?", rate_limiter=None)
    assert r.success
    assert r.output == pytest.approx(32.8)


def test_correccion_tipos_cambia_dtype(df, monkeypatch):
    monkeypatch.setattr(
        w, "chat_completion",
        _mock_fijo("df['edad'] = pd.to_numeric(df['edad'], errors='coerce')\nresultado = df"),
    )
    r = w.corregir_columnas_con_pandasai(df, "corregí edad a numérico", rate_limiter=None)
    assert r.success
    assert isinstance(r.output, pd.DataFrame)
    diff = w._diff_dtypes(df, r.output)
    assert "edad" in diff
    assert diff["edad"]["antes"] != diff["edad"]["despues"]
    # ciudad no se tocó -- no debería aparecer en el diff
    assert "ciudad" not in diff


def test_eliminacion_filas_baja_nunca_sube(df, monkeypatch):
    monkeypatch.setattr(
        w, "chat_completion",
        _mock_fijo("resultado = df.dropna(subset=['ciudad'])"),
    )
    r = w.eliminar_filas_con_pandasai(df, "eliminá las filas sin ciudad", rate_limiter=None)
    assert r.success
    assert len(r.output) < len(df)  # baja

    resumen = w.resumen_eliminacion_filas(df, r.output, "eliminá las filas sin ciudad")
    assert resumen["filas_antes"] == 5
    assert resumen["filas_despues"] == 4
    assert resumen["filas_eliminadas"] == 1
    assert resumen["criterio_aplicado"] == "eliminá las filas sin ciudad"
    assert len(resumen["df_eliminadas"]) == 1


def test_eliminacion_filas_rechaza_si_aumentan(df, monkeypatch):
    # código "eliminador" defectuoso que en realidad duplica filas
    monkeypatch.setattr(w, "chat_completion", _mock_fijo("resultado = pd.concat([df, df])"))
    r = w.eliminar_filas_con_pandasai(df, "pedido raro", rate_limiter=None)
    assert not r.success
    assert "aumentó el número de filas" in r.error


def test_grafico_devuelve_figure(df, monkeypatch):
    monkeypatch.setattr(
        w, "chat_completion",
        _mock_fijo(
            "fig, ax = plt.subplots()\n"
            "ax.bar(df['ciudad'].fillna('sin dato').astype(str), df['id'])\n"
            "resultado = fig"
        ),
    )
    r = w.generar_grafico_con_pandasai(df, "graficá id por ciudad", rate_limiter=None)
    assert r.success
    assert isinstance(r.output, Figure)


def test_grafico_rechaza_si_no_es_figure(df, monkeypatch):
    monkeypatch.setattr(
        w, "chat_completion",
        _mock_fijo("fig, ax = plt.subplots()\nresultado = ax"),  # error típico: ax en vez de fig
    )
    r = w.generar_grafico_con_pandasai(df, "graficá algo", rate_limiter=None)
    assert not r.success
    assert "no devolvió una Figure" in r.error


# ---------------------------------------------------------------------
# Rechazo por análisis estático de la sandbox -- NUNCA se reintenta
# ---------------------------------------------------------------------

def test_rechazo_por_sandbox_no_reintenta(df, monkeypatch):
    llamadas = {"n": 0}

    def mock_malicioso(mensajes, rate_limiter=None, tools=None, temperature=0.0):
        llamadas["n"] += 1
        return RespuestaFake("import os\nresultado = os.listdir('.')")

    monkeypatch.setattr(w, "chat_completion", mock_malicioso)
    r = w.analizar_con_pandasai(df, "algo raro", rate_limiter=None)
    assert not r.success
    assert r.rejected_by_static_analysis
    assert llamadas["n"] == 1  # una sola llamada -- no se le da la chance de "corregir" código malicioso


# ---------------------------------------------------------------------
# Error real de ejecución -- SÍ se reintenta (1 vez)
# ---------------------------------------------------------------------

def test_error_real_se_reintenta_una_vez(df, monkeypatch):
    llamadas = {"n": 0}

    def mock_falla_y_luego_corrige(mensajes, rate_limiter=None, tools=None, temperature=0.0):
        llamadas["n"] += 1
        if llamadas["n"] == 1:
            # columna inexistente -- error real de ejecución, no un rechazo de sandbox
            return RespuestaFake("resultado = df['columna_que_no_existe'].mean()")
        return RespuestaFake("resultado = df['id'].mean()")

    monkeypatch.setattr(w, "chat_completion", mock_falla_y_luego_corrige)
    r = w.analizar_con_pandasai(df, "promedio de algo", rate_limiter=None)
    assert r.success
    assert llamadas["n"] == 2  # falló una vez, se reintentó, funcionó
    assert r.output == 3.0


def test_error_real_agota_reintentos_y_falla(df, monkeypatch):
    llamadas = {"n": 0}

    def mock_siempre_falla(mensajes, rate_limiter=None, tools=None, temperature=0.0):
        llamadas["n"] += 1
        return RespuestaFake("resultado = df['columna_que_no_existe'].mean()")

    monkeypatch.setattr(w, "chat_completion", mock_siempre_falla)
    r = w.analizar_con_pandasai(df, "promedio de algo", rate_limiter=None)
    assert not r.success
    assert not r.rejected_by_static_analysis
    # 1 intento inicial + _MAX_REINTENTOS_POR_ERROR (1) = 2 llamadas totales, nunca más
    assert llamadas["n"] == 1 + w._MAX_REINTENTOS_POR_ERROR