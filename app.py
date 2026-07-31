"""
app.py
Interfaz de Streamlit de DataQualityAgent.
"""
import streamlit as st
from pathlib import Path

from config import APP_TITLE, APP_ICON, APP_LAYOUT, DATA_FILE_TYPES
from ui.state import StateManager
from ui import diagnostico_tab, chat_tab, herramientas_tab, diccionario_tab, analisis_tab

# ── Funciones de Utilidad y Componentes (Views) ──────────────

def load_css(file_path: str):
    """Carga y aplica un archivo CSS externo usando pathlib."""
    path = Path(file_path)
    if path.is_file():
        st.markdown(f"<style>{path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"No se encontró el archivo de estilos: {file_path}")

def render_header():
    """Renderiza el título y la tarjeta descriptiva de la aplicación."""
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.markdown(
        """
        <div class="hero-card">
            <strong>Agente de data wrangling</strong> para limpiar y transformar tus datos
            de dos formas: por <strong>chat</strong> (lenguaje natural) o de forma
            <strong>manual</strong> con herramientas precisas.
        </div>
        """,
        unsafe_allow_html=True
    )

# ── Función Principal (Controller) ───────────────────────────

def main():
    # 1. Configuración inicial de la página
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout=APP_LAYOUT)
    
    # 2. Carga de estilos (Separación de UI y Lógica)
    load_css("assets/style.css") # Ajusta la ruta según tu estructura de carpetas
    
    # 3. Renderizar Header
    render_header()

    # 4. Inicializar Estado (Model/State)
    manager = StateManager()
    manager.ensure_initialized()

    # 5. Interfaz de carga de archivos
    archivo = st.file_uploader("📂 Subí tu archivo de datos", type=DATA_FILE_TYPES)
    manager.handle_file_upload(archivo)

    # 6. Barrera de validación (Early Return)
    if not manager.has_session():
        st.info(
            "👆 Subí un Excel o CSV para empezar.\n\n"
            "Luego podrás diagnosticar la calidad de los datos, limpiarlos por chat "
            "o usar las herramientas manuales."
        )
        return  # Reemplaza st.stop() por return dentro de una función

    # 7. Renderizar Sidebar y Tabs
    manager.render_sidebar()

    tab_diagnostico, tab_chat, tab_analisis, tab_herramientas, tab_diccionario = st.tabs(
        ["📋 Diagnóstico", "💬 Chat", "🧮 Análisis libre", "🔧 Herramientas", "📖 Diccionario"]
    )

    with tab_diagnostico:
        diagnostico_tab.render(manager)

    with tab_chat:
        chat_tab.render(manager)

    with tab_analisis:
        analisis_tab.render(manager)

    with tab_herramientas:
        herramientas_tab.render(manager)

    with tab_diccionario:
        diccionario_tab.render(manager)


# ── Punto de Entrada ─────────────────────────────────────────
if __name__ == "__main__":
    main()