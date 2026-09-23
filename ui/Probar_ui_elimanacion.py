import sys
sys.path.insert(0, "/home/claude/dqa")

import pandas as pd
from streamlit.testing.v1 import AppTest

import analysis.pandasai_wrapper as w


class RespuestaFake:
    def __init__(self, texto):
        self.texto = texto


CODIGO_ELIMINACION = "resultado = df.dropna(subset=['ciudad'])"


def mock_chat_completion(mensajes, rate_limiter=None, tools=None, temperature=0.0):
    return RespuestaFake(CODIGO_ELIMINACION)


w.chat_completion = mock_chat_completion
w.extract_text = lambda r: r.texto

app_script = """
import sys
sys.path.insert(0, "/home/claude/dqa")

import streamlit as st
import pandas as pd

from core.memory import create_session
from ui.state import StateManager
from ui import analisis_tab

manager = StateManager()
manager.ensure_initialized()

if not manager.has_session():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "ciudad": ["SP", None, "RJ", None, "BH"],
    })
    sesion = create_session(df)
    manager.set_session(sesion)

analisis_tab.render(manager)
"""

with open("/home/claude/dqa/_app_test_script2.py", "w") as f:
    f.write(app_script)

at = AppTest.from_file("/home/claude/dqa/_app_test_script2.py")
at.run()
assert not at.exception

print("--- 1. Pido eliminación de filas sin ciudad ---")
at.chat_input[0].set_value("eliminá las filas sin ciudad").run()
assert not at.exception, f"Excepción: {at.exception}"

botones = [b.label for b in at.button]
print("botones:", botones)
assert "✅ Aplicar eliminación" in botones
assert "❌ Descartar" in botones

sesion_antes = at.session_state["session"]
assert len(sesion_antes.current_df) == 5  # todavía sin aplicar
assert len(sesion_antes.actions_log) == 0

print("--- 2. Confirmo la eliminación ---")
at.button(key="aplicar_eliminacion_analisis").click().run()
assert not at.exception, f"Excepción al confirmar: {at.exception}"

sesion_despues = at.session_state["session"]
print("filas después:", len(sesion_despues.current_df))
assert len(sesion_despues.current_df) == 3  # 5 -> 3 (se sacaron las 2 con ciudad nula)
assert len(sesion_despues.actions_log) == 1
log = sesion_despues.actions_log[0]
print("actions_log:", log)
assert log["tool_name"] == "pandasai_eliminacion_filas"
assert log["result_summary"]["filas_antes"] == 5
assert log["result_summary"]["filas_despues"] == 3
assert log["result_summary"]["filas_eliminadas"] == 2
assert "df_eliminadas" not in log["result_summary"]  # no se loguea el DataFrame completo
assert len(sesion_despues.df_history) == 1

print("\\nTODO OK -- flujo de eliminación funciona de punta a punta")