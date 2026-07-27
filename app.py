"""
app.py

Interfaz de Streamlit de DataQualityAgent.

Orquestador delgado: delega toda la lógica de cada sección a los
módulos de ui/ (state, diagnostico_tab, chat_tab, herramientas_tab,
diccionario_tab).

Cuatro secciones:
  1. Diagnóstico — al subir el archivo, corre automáticamente y muestra
     qué hace falta limpiar (y qué NO), con un score de calidad.
  2. Chat — lenguaje natural, el agente decide qué tool(s) usar.
  3. Herramientas manuales — un input dedicado para cada una de las 15
     tools, para control preciso sin depender del LLM.
  4. Diccionario de datos — subir un documento (PDF/Word/Excel/CSV) que
     describa las columnas, para que el chat lo use como contexto (RAG).

Corré con: streamlit run app.py
"""

import streamlit as st

from config import APP_TITLE, APP_ICON, APP_LAYOUT, DATA_FILE_TYPES
from ui.state import StateManager
from ui import diagnostico_tab, chat_tab, herramientas_tab, diccionario_tab

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout=APP_LAYOUT)

# ── State manager ────────────────────────────────────────────
manager = StateManager()
manager.ensure_initialized()

# ── Header ───────────────────────────────────────────────────
st.title(f"{APP_ICON} {APP_TITLE}")
st.caption("Subí un Excel o CSV y dejá que el agente te diga qué falta limpiar — o hacelo vos mismo, a mano.")

# ── File upload ──────────────────────────────────────────────
archivo = st.file_uploader("📂 Subí tu archivo de datos", type=DATA_FILE_TYPES)
manager.handle_file_upload(archivo)

if not manager.has_session():
    st.info("👆 Subí un archivo para empezar.")
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────
manager.render_sidebar()

# ── Tabs ─────────────────────────────────────────────────────
tab_diagnostico, tab_chat, tab_herramientas, tab_diccionario = st.tabs(
    ["📋 Diagnóstico", "💬 Chat", "🔧 Herramientas manuales", "📖 Diccionario de datos"]
)

with tab_diagnostico:
    diagnostico_tab.render(manager)

with tab_chat:
    chat_tab.render(manager)

with tab_herramientas:
    herramientas_tab.render(manager)

with tab_diccionario:
    diccionario_tab.render(manager)