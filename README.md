# DataQualityAgent

## 📋 Descripción General

**DataQualityAgent** es una aplicación construida en Streamlit que automatiza la limpieza, el control de calidad y el análisis exploratorio de datasets, combinando un motor determinístico basado en pandas con capacidades de lenguaje natural impulsadas por LLM.

El sistema permite a un usuario cargar un archivo de datos y, mediante una interfaz conversacional o herramientas guiadas, ejecutar tareas como:

- Diagnóstico de calidad de datos (nulos, duplicados, tipos, outliers).
- Limpieza y estandarización de columnas (normalización, binning, formateo).
- Corrección de datos y eliminación de filas mediante instrucciones en lenguaje natural.
- Trazabilidad completa de cada transformación aplicada sobre el dataset.

El diseño prioriza la seguridad: el modelo de lenguaje **nunca ejecuta código arbitrario** directamente. Toda acción pasa por un despachador de herramientas (*tool dispatcher*) que solo puede invocar funciones predefinidas y auditadas, o bien por un entorno aislado (*sandbox*) cuando se requiere generación dinámica de código.

---

## 🏗️ Arquitectura de la Solución

El proyecto está organizado en módulos con responsabilidades claramente separadas:

```
DataQualityAgent/
├── app.py                     # Punto de entrada de la aplicación Streamlit
├── core/
│   ├── agent.py                # Orquestador principal (run_diagnosis, send_message)
│   └── memory.py                # SessionMemory: estado de sesión y trazabilidad
├── cleaning/
│   ├── missing_values.py        # Manejo de valores faltantes
│   ├── formatting.py            # Formateo de columnas
│   ├── standardization.py       # Estandarización (z-score)
│   ├── normalization.py         # Normalización (min-max)
│   └── binning.py               # Binning (equal_width, equal_frequency, custom)
├── quality/
│   └── profiler.py              # Motor de perfilado determinístico (pandas puro, sin LLM)
├── analysis/
│   └── pandasai_wrapper.py      # Integración con PandasAI para correcciones vía LLM
├── security/
│   └── sandbox.py               # Entorno aislado para ejecución de código generado
├── rag/
│   ├── ingest.py                 # Ingesta de documentos
│   ├── vectorstore.py            # Almacenamiento vectorial
│   └── validator.py              # Validación de contenido recuperado
├── tools/
│   └── tool_definitions.py       # Definición de herramientas (schema tipo OpenAI) y dispatch_tool()
└── ui/
    ├── analisis_tab.py            # Tab principal de análisis libre
    └── state.py                   # Manejo de estado de la interfaz
```

### Flujo de orquestación

1. El usuario interactúa desde la interfaz de Streamlit (carga de archivo, preguntas, solicitudes de limpieza).
2. `core/agent.py` recibe la solicitud y decide qué herramienta ejecutar.
3. Toda acción sobre el dataset pasa por `dispatch_tool()`, que solo puede invocar el conjunto fijo de funciones ya escritas y probadas en `tools/tool_definitions.py`.
4. Para tareas que requieren generación dinámica de código (por ejemplo, corregir una columna con una instrucción ambigua en lenguaje natural), se utiliza `analysis/pandasai_wrapper.py`, cuyo código se ejecuta exclusivamente dentro de `security/sandbox.py`.
5. Cada modificación al dataset se registra a través de `record_action()` en `core/memory.py`, actualizando el DataFrame actual y manteniendo un historial completo (`df_history`) y un log de acciones (`actions_log`).

### Manejo de estado

`SessionMemory` centraliza el estado de la sesión:

- `current_df` / `original_df`: dataset en su estado actual y original.
- `df_history`: historial de versiones del dataset tras cada transformación.
- `actions_log`: registro auditable de cada acción ejecutada.
- `rate_limiter`: control de frecuencia de llamadas.

Esto garantiza trazabilidad total: en cualquier momento se puede saber qué se cambió, cuándo y con qué herramienta.

---

## 🛠️ Tecnologías y Herramientas Utilizadas

| Categoría | Tecnología |
|---|---|
| Interfaz de usuario | Streamlit |
| Procesamiento de datos | pandas |
| Orquestación LLM | LiteLLM (compatible con Groq) |
| Generación de código dinámico | PandasAI |
| Definición de herramientas | Schema estilo OpenAI (tool calling) |
| Recuperación aumentada (RAG) | Vectorstore propio + validadores |
| Testing | `streamlit.testing.v1.AppTest` |
| Lenguaje | Python |

---

## 🚀 Instrucciones para Ejecutar el Proyecto

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd DataQualityAgent
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv .venv
source .venv/bin/activate      # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configurar variables de entorno

Crear un archivo `.env` en la raíz del proyecto con las claves necesarias (por ejemplo, API key del proveedor LLM usado vía LiteLLM/Groq):

```
GROQ_API_KEY=tu_api_key_aqui
```

### 4. Ejecutar la aplicación

```bash
streamlit run app.py
```

### 5. Ejecutar los tests (opcional)

```bash
pytest
```

---

## 💬 Ejemplos de Preguntas que el Agente Puede Responder

- "¿Cuántos valores nulos tiene cada columna?"
- "¿Hay filas duplicadas en este dataset?"
- "Rellená los valores nulos de la columna 'edad' con 0."
- "Eliminá las filas donde la columna 'email' esté vacía."
- "Normalizá la columna 'ingresos' usando min-max."
- "Aplicá binning de frecuencia igual a la columna 'edad' en 5 grupos."
- "Corregí el formato de la columna 'fecha' a formato estándar."
- "Mostrame un resumen de calidad general del dataset."

---

## 📝 Ejemplos de Respuestas Generadas por el Agente

**Pregunta:** *"¿Cuántos valores nulos tiene cada columna?"*

**Respuesta:**
```
Se detectaron valores nulos en las siguientes columnas:
- edad: 12 valores nulos (3.4%)
- email: 5 valores nulos (1.4%)
- ciudad: 0 valores nulos
El resto de las columnas no presenta valores faltantes.
```

**Pregunta:** *"Rellená los valores nulos de la columna 'edad' con 0."*

**Respuesta:**
```
✅ Operación completada: se reemplazaron 12 valores nulos en la columna 'edad' por 0.
El cambio quedó registrado en el historial de acciones.
```

**Pregunta:** *"¿Hay filas duplicadas?"*

**Respuesta:**
```
Se encontraron 7 filas completamente duplicadas en el dataset.
¿Deseas que las elimine o que te muestre cuáles son antes de proceder?
```

---

## 🖼️ Capturas del Proyecto en Curso

*(E<img width="875" height="981" alt="image9" src="https://github.com/user-attachments/assets/38cdfeeb-65f6-4548-be21-ea899cdce19a" />
spacio reservado para agregar imágenes/screenshots del proyecto)*

<img width="908" height="1049" alt="image11" src="https://github.com/user-attachments/assets/d223bcd2-55e2-4e9b-ade7-00c7eb7c85ee" />

<img width="907" height="847" alt="image15" src="https://github.com/user-attachments/assets/2434f3d2-e18f-4214-aa91-19609f6cc8c6" />
<img width="960" height="1175" alt="image14" src="https://github.com/user-attachments/assets/b40dfc6d-969b-47c5-a3ea-4b4eff26f2cd" />

<!--
![Diagnóstico inicial](ruta/a/imagen1.png)
![Tab de análisis](ruta/a/imagen2.png)
![Ejemplo de limpieza aplicada](ruta/a/imagen3.png)
-->docs: agregar README completo del proyecto.
