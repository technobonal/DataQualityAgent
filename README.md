# 🧹 DataQualityAgent

Agente de calidad de datos con interfaz conversacional. Subí un archivo CSV o Excel y el agente te dice qué falta limpiar, con un score de calidad — o chateá en lenguaje natural para que lo haga por vos.

---

## ✨ Features

- **Diagnóstico automático** — al subir el archivo, corre un análisis completo y muestra qué hace falta limpiar (y qué NO), con un score de 0 a 100.
- **Chat en lenguaje natural** — pedile al agente que limpie tus datos como si hablaras con un compañero de equipo. El agente decide qué herramienta(s) usar.
- **15 herramientas de limpieza** — valores faltantes, duplicados, outliers, formato de datos, nombres de columnas, texto, estandarización, normalización, binning, one-hot encoding, y más.
- **Control manual** — si preferís, ejecutá cada herramienta directamente desde un panel visual, sin pasar por el chat.
- **Diccionario de datos (RAG)** — subí un PDF/Word/Excel/CSV que explique las columnas y el chat lo usa como contexto.
- **Undo/Reset** — deshacé cualquier paso o volvé al archivo original en cualquier momento.
- **Seguridad** — detección de prompt injection en los datos, rate limiting, sandbox para código generado.

---

## 🛠️ Instalación

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd DataqualityAgent

# 2. Crear y activar entorno virtual
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
copy .env.example .env
# Editá .env con tus API keys (ver abajo)
```

### API Keys necesarias

| Servicio | Para qué se usa | Dónde conseguirla |
|---|---|---|
| **Groq** (obligatoria) | LLM principal (Llama 3.3 70B) | [console.groq.com/keys](https://console.groq.com/keys) |
| **Cohere** (opcional) | Embeddings para el RAG del diccionario de datos | [dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys) |
| **Hugging Face** (opcional) | Detección de datos sensibles | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

### Modo local (sin Groq)

Si preferís no usar Groq (o no tenés API key), podés usar Ollama local:

```bash
# Instalar Ollama: https://ollama.com
ollama pull qwen2.5:7b
ollama serve
```

En el `.env`, poné `USE_LOCAL=true`.

---

## 🚀 Uso

```bash
streamlit run app.py
```

Se abre en `http://localhost:8501`. Subí un CSV o Excel y empezá a trabajar.

---

## 📁 Arquitectura

```
DataqualityAgent/
├── app.py                  # Orquestador Streamlit (delega a ui/)
├── config.py               # Configuración centralizada
├── ui/                     # Capa de presentación
│   ├── state.py            #   StateManager (session state)
│   ├── components.py       #   Widgets reutilizables
│   ├── diagnostico_tab.py  #   Tab: Diagnóstico
│   ├── chat_tab.py         #   Tab: Chat
│   ├── herramientas_tab.py #   Tab: Herramientas manuales
│   └── diccionario_tab.py  #   Tab: Diccionario de datos
├── core/                   # Lógica central
│   ├── agent.py            #   Orquestador del agente (LLM + tools)
│   ├── memory.py           #   Memoria de sesión (historial, undo)
│   └── llm_client.py       #   Cliente LLM (Groq + Ollama fallback)
├── tools/
│   └── tool_definitions.py #   Schemas + dispatcher de las 15 tools
├── cleaning/               # Implementación de tools de limpieza
│   ├── missing_values.py   #   Valores faltantes
│   ├── duplicates.py       #   Duplicados
│   ├── formatting.py       #   Formato de datos + nombres + texto
│   ├── standardization.py  #   Estandarización z-score
│   ├── normalization.py    #   Normalización min-max
│   ├── binning.py          #   Discretización
│   └── indicators.py       #   Variables indicadoras (one-hot)
├── quality/                # Análisis de calidad
│   ├── outliers.py         #   Detección y tratamiento de outliers
│   ├── report.py           #   Reporte de calidad
│   └── recommendations.py  #   Recomendaciones automáticas
├── security/               # Seguridad
│   ├── validators.py       #   Detección de prompt injection
│   ├── rate_limiter.py     #   Rate limiting para Groq
│   └── sandbox.py          #   Sandbox para código generado
├── rag/                    # Retrieval-Augmented Generation
│   ├── ingest.py           #   Extracción de documentos
│   ├── validator.py        #   Validación y filtrado
│   └── vectorstore.py      #   ChromaDB vector store
└── tests/                  # Tests
```

---

## 🔧 Stack técnico

- **LLM**: Groq (Llama 3.3 70B) vía LiteLLM, con fallback a Ollama local
- **Interfaz**: Streamlit
- **Datos**: Pandas, NumPy
- **RAG**: ChromaDB + Cohere embeddings
- **Seguridad**: Detección de prompt injection, rate limiting, sandbox

---

## 📝 Licencia

MIT
