"""
security/rate_limiter.py

Limitador de tasa (rate limiter) del lado del cliente: evita que el
agente le pegue a Groq con más tokens por minuto de los que el free tier
permite, en vez de descubrirlo recién cuando Groq devuelve un 429
(como pasó durante las pruebas del proyecto — ver core/llm_client.py).

Esto NO es sobre autenticación/autorización — es sobre protección de
cuota: una sesión no debería poder "quemar" sin querer los tokens/minuto
disponibles pidiendo demasiadas cosas seguidas.

Sin dependencia de un LLM en sí — solo lleva la cuenta de tokens usados
en una ventana de tiempo deslizante. litellm se usa únicamente para
ESTIMAR cuántos tokens va a pesar una llamada antes de hacerla (no para
llamar al modelo).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

try:
    import litellm
    _TIENE_LITELLM = True
except ImportError:
    _TIENE_LITELLM = False

# Límite conservador por defecto: el free tier de Groq para
# llama-3.3-70b-versatile es 12.000 tokens/min (ver core/llm_client.py).
# Se usa un margen de seguridad del 80% para no pegarle justo al límite
# real y terminar en el mismo 429 que se quiere evitar.
_DEFAULT_TPM_LIMIT = 12000
_DEFAULT_SAFETY_MARGIN = 0.8
_WINDOW_SECONDS = 60


@dataclass
class RateLimiter:
    """
    Rate limiter de ventana deslizante: lleva la cuenta de tokens
    consumidos en los últimos window_seconds segundos, y decide si una
    nueva llamada "entra" en el presupuesto sin pasarse.

    Un RateLimiter por sesión de Streamlit alcanza para este proyecto
    (no es necesario compartirlo entre usuarios — cada sesión tiene su
    propio presupuesto, igual que cada sesión tiene su propia SessionMemory
    en core/memory.py).
    """
    tpm_limit: int = _DEFAULT_TPM_LIMIT
    safety_margin: float = _DEFAULT_SAFETY_MARGIN
    window_seconds: int = _WINDOW_SECONDS
    _historial: deque = field(default_factory=deque, repr=False)  # [(timestamp, tokens), ...]

    @property
    def presupuesto_efectivo(self) -> int:
        """Tokens/min disponibles después de aplicar el margen de seguridad."""
        return int(self.tpm_limit * self.safety_margin)

    def _limpiar_historial(self) -> None:
        """Descarta del historial las entradas más viejas que la ventana."""
        ahora = time.monotonic()
        while self._historial and (ahora - self._historial[0][0]) > self.window_seconds:
            self._historial.popleft()

    def tokens_usados_en_ventana(self) -> int:
        """Cuántos tokens se consumieron en la ventana de tiempo actual."""
        self._limpiar_historial()
        return sum(tokens for _, tokens in self._historial)

    def puede_proceder(self, tokens_estimados: int) -> bool:
        """¿Esta llamada, sumada a lo ya usado en la ventana, entra en el presupuesto?"""
        return (self.tokens_usados_en_ventana() + tokens_estimados) <= self.presupuesto_efectivo

    def tiempo_de_espera(self) -> float:
        """
        Cuántos segundos hay que esperar hasta que se libere lugar en el
        presupuesto (0.0 si ya se puede proceder ahora mismo).
        """
        self._limpiar_historial()
        if not self._historial:
            return 0.0
        entrada_mas_vieja_timestamp = self._historial[0][0]
        segundos_transcurridos = time.monotonic() - entrada_mas_vieja_timestamp
        return max(0.0, self.window_seconds - segundos_transcurridos)

    def registrar_uso(self, tokens: int) -> None:
        """Registra que se consumieron `tokens` tokens ahora mismo."""
        self._historial.append((time.monotonic(), tokens))
        self._limpiar_historial()


def estimar_tokens(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    model: str = "groq/llama-3.3-70b-versatile",
) -> int:
    """
    Estima cuántos tokens va a consumir una llamada al LLM, ANTES de
    hacerla — para que RateLimiter pueda decidir si conviene esperar.

    Usa litellm.token_counter() (cuenta real según el tokenizer del
    modelo, incluyendo el peso de los tool schemas); si litellm no está
    disponible por algún motivo, cae a una aproximación simple
    (~4 caracteres por token, razonable tanto para español como inglés).
    """
    if _TIENE_LITELLM:
        try:
            return litellm.token_counter(model=model, messages=messages, tools=tools)
        except Exception:
            pass  # cae a la aproximación simple

    texto_total = " ".join(str(m.get("content") or "") for m in messages)
    if tools:
        import json
        texto_total += " " + json.dumps(tools, ensure_ascii=False)
    return max(1, len(texto_total) // 4)


if __name__ == "__main__":
    # Ejemplo rápido para probar el módulo de forma aislada, simulando
    # varias llamadas seguidas hasta agotar el presupuesto.
    limiter = RateLimiter(tpm_limit=12000, safety_margin=0.8)  # presupuesto real: 9600

    print("Presupuesto efectivo:", limiter.presupuesto_efectivo, "tokens/min")

    mensajes_ejemplo = [
        {"role": "system", "content": "Sos un agente de calidad de datos." * 20},
        {"role": "user", "content": "Rellename los nulos de edad con la media"},
    ]
    tokens_estimados = estimar_tokens(mensajes_ejemplo)
    print(f"Tokens estimados para un mensaje típico: {tokens_estimados}")

    print("\n--- Simulamos llamadas seguidas hasta que el presupuesto no alcance ---")
    intento = 1
    while limiter.puede_proceder(tokens_estimados) and intento <= 5:
        limiter.registrar_uso(tokens_estimados)
        print(f"Llamada {intento}: OK. Usado en ventana: {limiter.tokens_usados_en_ventana()}")
        intento += 1

    if not limiter.puede_proceder(tokens_estimados):
        print(f"Llamada {intento}: RECHAZADA — esperar {limiter.tiempo_de_espera():.1f}s antes de reintentar.")