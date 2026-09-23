"""
Router de intenciones para el análisis conversacional (pandasai_wrapper).

Cascada:
  1. Reglas (keywords)
  2. LLM JSON (solo si las reglas no resuelven)
  3. ambiguo → la UI pregunta
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from core.llm_client import chat_completion, extract_text, LLMClientError
from security.rate_limiter import RateLimiter

IntentName = Literal[
    "analizar",
    "corregir_tipos",
    "eliminar_filas",
    "eliminar_columnas",
    "limpiar_celdas",
    "grafico",
    "ambiguo",
]

INTENTS: tuple[IntentName, ...] = (
    "analizar",
    "corregir_tipos",
    "eliminar_filas",
    "eliminar_columnas",
    "limpiar_celdas",
    "grafico",
    "ambiguo",
)

CONFIDENCE_MINIMA = 0.7


@dataclass(frozen=True)
class IntentResult:
    intent: IntentName
    confidence: float
    method: Literal["rules", "llm", "fallback"]
    reason: str = ""


_RE_GRAFICO = re.compile(
    r"\b(gr[aá]fico|grafico|plot|plote|barras?|histograma|scatter|l[ií]nea|pie|torta)\b",
    re.I,
)
_RE_CELDAS = re.compile(
    r"\b(rellen\w*|fillna|imput\w*|vaci(?:ar|á|a)\s+celdas?|"
    r"poner\s+(la\s+)?(media|mediana|cero)|reemplaz\w*\s+nulos?)\b",
    re.I,
)
_RE_TIPOS = re.compile(
    r"\b(datetime|to_numeric|to_datetime|dtype|caste\w*|convert(?:ir|í|i)|"
    r"tipo(s)?\s+de\s+dato|como\s+fecha|a\s+n[uú]mero|a\s+num[eé]rico)\b",
    re.I,
)
_RE_ELIM = re.compile(
    r"\b(elimin\w*|borr\w*|sac(?:ar|á|a)|quit(?:ar|á|a)|drop|remover?|tir(?:ar|á|a))\b",
    re.I,
)
_RE_COL = re.compile(r"\bcolumnas?\b", re.I)
_RE_FIL = re.compile(r"\bfilas?\b", re.I)
_RE_DUPLIC = re.compile(r"\bduplicad\w*\b", re.I)
_RE_ANALISIS = re.compile(
    r"\b(promedio|media|mediana|suma|cu[aá]ntos?|cu[aá]ntas?|filtr\w*|"
    r"groupby|agrup\w*|top\s*\d+|m[aá]ximo|m[ií]nimo|listame|mostrame|"
    r"cu[aá]l\s+es|qu[eé]\s+columnas?|porcentaje|nulos?\s+por)\b",
    re.I,
)


def clasificar_por_reglas(texto: str) -> IntentResult | None:
    t = (texto or "").strip()
    if not t:
        return IntentResult("ambiguo", 0.0, "rules", "texto vacío")

    if _RE_GRAFICO.search(t):
        return IntentResult("grafico", 0.95, "rules", "keyword gráfico")

    if _RE_CELDAS.search(t):
        return IntentResult("limpiar_celdas", 0.9, "rules", "keyword relleno/vaciado celdas")

    if _RE_TIPOS.search(t):
        return IntentResult("corregir_tipos", 0.9, "rules", "keyword tipos/datetime/numeric")

    habla_elim = bool(_RE_ELIM.search(t))
    habla_col = bool(_RE_COL.search(t))
    habla_fil = bool(_RE_FIL.search(t))
    habla_dup = bool(_RE_DUPLIC.search(t))

    if habla_dup and not habla_col:
        return IntentResult("eliminar_filas", 0.92, "rules", "duplicados → filas")

    if habla_elim and habla_col and habla_fil:
        return IntentResult("ambiguo", 0.4, "rules", "menciona filas y columnas")

    if habla_elim and habla_col and not habla_fil:
        return IntentResult("eliminar_columnas", 0.93, "rules", "eliminar + columna(s)")

    if habla_elim and habla_fil and not habla_col:
        return IntentResult("eliminar_filas", 0.93, "rules", "eliminar + fila(s)")

    if habla_elim and re.search(r"\bnulos?\b", t, re.I) and not habla_col:
        return IntentResult("eliminar_filas", 0.75, "rules", "eliminar nulos (asume filas)")

    if habla_elim and not habla_col and not habla_fil:
        return None

    if _RE_ANALISIS.search(t) and not habla_elim:
        return IntentResult("analizar", 0.8, "rules", "keyword análisis")

    return None


_PROMPT_ROUTER = """\
Clasificá el pedido del usuario sobre un dataset tabular en UNA intención.

Intenciones válidas:
- analizar: solo lectura (promedios, conteos, filtros, listados, "¿cuántos?")
- corregir_tipos: cambiar dtype (fechas, números, booleanos)
- eliminar_filas: borrar filas (nulos, duplicados, filtros de filas)
- eliminar_columnas: borrar columnas enteras
- limpiar_celdas: rellenar o vaciar celdas (fillna, media, mediana, cero) sin borrar filas/columnas
- grafico: crear un gráfico
- ambiguo: no se puede decidir con seguridad, o pide varias acciones a la vez

{contexto}
Pedido del usuario: {texto}

Reglas:
- Si menciona "columna(s)" y eliminar/sacar/tirar → eliminar_columnas
- Si menciona "fila(s)" o duplicados y eliminar → eliminar_filas
- Si pide rellenar/imputar/vaciar valores → limpiar_celdas
- Si es una pregunta de información sin mutar datos → analizar
- Si el pedido es vago ("limpiá el dataset") → ambiguo
- Respondé SOLO un JSON válido, sin markdown, con esta forma exacta:
{{"intent": "<una de las intenciones>", "confidence": <0.0 a 1.0>, "reason": "<muy breve>"}}
"""


def _parse_llm_intent(raw: str) -> IntentResult:
    texto = (raw or "").strip()
    texto = re.sub(r"^```(?:json)?\s*", "", texto)
    texto = re.sub(r"\s*```$", "", texto)
    try:
        data = json.loads(texto)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", texto)
        if not m:
            return IntentResult("ambiguo", 0.0, "llm", "JSON inválido")
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return IntentResult("ambiguo", 0.0, "llm", "JSON inválido")

    intent = str(data.get("intent", "ambiguo")).strip().lower()
    if intent not in INTENTS:
        intent = "ambiguo"
    try:
        conf = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    reason = str(data.get("reason", ""))[:200]
    return IntentResult(intent, conf, "llm", reason)  # type: ignore[arg-type]


def clasificar_con_llm(
    texto: str,
    rate_limiter: RateLimiter,
    contexto_previo: str | None = None,
) -> IntentResult:
    ctx = ""
    if contexto_previo:
        ctx = (
            "Contexto de la conversación (puede resolver pronombres como "
            f"'esas', 'esos'):\n{contexto_previo}\n"
        )
    prompt = _PROMPT_ROUTER.format(texto=texto.strip(), contexto=ctx)
    try:
        respuesta = chat_completion(
            [{"role": "user", "content": prompt}],
            rate_limiter=rate_limiter,
            tools=None,
            temperature=0.0,
        )
        raw = extract_text(respuesta) or ""
    except LLMClientError as e:
        return IntentResult("ambiguo", 0.0, "fallback", f"LLM falló: {e}")
    return _parse_llm_intent(raw)


def clasificar_intent(
    texto: str,
    rate_limiter: RateLimiter,
    contexto_previo: str | None = None,
    *,
    forzar_llm: bool = False,
) -> IntentResult:
    if not forzar_llm:
        por_reglas = clasificar_por_reglas(texto)
        if por_reglas is not None and por_reglas.intent != "ambiguo":
            return por_reglas
        if por_reglas is not None and por_reglas.intent == "ambiguo":
            return por_reglas
    return clasificar_con_llm(texto, rate_limiter, contexto_previo=contexto_previo)


def necesita_aclaracion(result: IntentResult, umbral: float = CONFIDENCE_MINIMA) -> bool:
    return result.intent == "ambiguo" or result.confidence < umbral