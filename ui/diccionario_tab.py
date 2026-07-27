"""
ui/diccionario_tab.py

Tab de Diccionario de datos (RAG) — permite subir un documento
(PDF, Word, Excel, CSV) que describe las columnas del dataset.
El chat lo usa como contexto adicional vía RAG.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from core.agent import ingest_data_dictionary
from ui.state import StateManager


def render(manager: StateManager) -> None:
    """Renderiza la pestaña de Diccionario de datos."""
    st.caption(
        "Subí un documento que explique qué significa cada columna de tu dataset "
        "(PDF, Word, Excel o CSV) — el chat lo va a usar como contexto adicional."
    )

    dic_file = st.file_uploader(
        "📖 Diccionario de datos",
        type=["pdf", "docx", "csv", "xlsx"],
        key="dic_uploader",
    )

    if dic_file is not None:
        if st.button("Procesar e indexar"):
            extension = Path(dic_file.name).suffix
            with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp:
                tmp.write(dic_file.getvalue())
                ruta_temporal = tmp.name

            with st.spinner("Procesando el diccionario de datos…"):
                try:
                    resultado = ingest_data_dictionary(ruta_temporal)
                except Exception as e:
                    st.error(f"No se pudo procesar el documento: {e}")
                    return

            if resultado["validacion"]["is_valid"]:
                st.success(
                    f"Documento indexado: {resultado['guardado']['chunks_added']} fragmento(s) guardados."
                )
            else:
                st.error(
                    "El documento no pasó la validación: "
                    + "; ".join(resultado["validacion"]["issues"])
                )