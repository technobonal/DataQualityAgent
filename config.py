"""
config.py

Configuración centralizada de DataQualityAgent.
Constantes, rutas y feature flags que antes estaban dispersos en varios módulos.
"""

from __future__ import annotations

import os
import logging

from dotenv import load_dotenv

load_dotenv()

# ─── Logging ─────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s │ %(name)-25s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dataqualityagent")

# ─── Paths ───────────────────────────────────────────────────
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

# ─── Agent defaults ─────────────────────────────────────────
MAX_TOOL_ITERATIONS = 5     # Max LLM↔tool round-trips antes de cortar
MAX_HISTORY_MESSAGES = 20   # Mensajes de historial enviados al LLM (sin system)
MAX_UNDO_STEPS = 10         # Snapshots guardados para undo
MAX_ITEMS_EN_LISTA_PARA_LLM = 10  # Listas truncadas en resultados para el LLM

# ─── Streamlit ───────────────────────────────────────────────
APP_TITLE = "DataQualityAgent"
APP_ICON = "🧹"
APP_LAYOUT = "wide"

# ─── Supported file types ───────────────────────────────────
DATA_FILE_TYPES = ["csv", "xlsx", "xls"]
DICTIONARY_FILE_TYPES = ["pdf", "docx", "csv", "xlsx"]
