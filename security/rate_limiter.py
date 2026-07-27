"""
security/rate_limiter.py

Limitador de tasa (rate limiter) del lado del cliente: evita que el
agente le pegue a Groq con más tokens por minuto de los que cada modelo
de la cadena permite, en vez de descubrirlo recién cuando Groq devuelve
un 429 (como pasó durante las pruebas del proyecto — ver
core/llm_client.py).

REDISEÑO: cada modelo de GROQ_MODEL_CHAIN tiene su PROPIA cuota de
tokens/minuto separada en Groq (no un pool único compartido):

    llama-3.3-70b-versatile   -> 12.000 TPM
    openai/gpt-oss-20b        ->  8.000 TPM
    openai/gpt-oss-120b       ->  8.000 TPM
    llama-3.1-8b-instant      ->  6.000 TPM
    -----------------------------------------
    Total combinado           -> 34.000 TPM

Tratar esto como un pool único de 34.000 (versión anterior de este
archivo) subestima el presupuesto real disponible: si el modelo 1 se
queda sin cuota, los otros tres siguen teniendo la suya intacta. Por
eso ahora se lleva la cuenta POR MODELO, en paralelo a como
core/llm_client.py ya recorre la cadena modelo por modelo.

Esto NO es sobre autenticación/autorización — es sobre protección de
cuota: una sesión no debería poder "quemar" sin querer los tokens/minuto
disponibles de un modelo pidiendo demasiadas cosas seguidas.

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

# Cuota TPM real por modelo, tal como está documentada en la cadena de
# core/llm_client.py (GROQ_MODEL_CHAIN). Si tu cuenta de Groq cambia de
# plan y estos números cambian, actualizar acá.
MODEL_TPM_LIMITS: dict[str, int] = {
    "llama-3.3-70b-versatile": 12000,
    "openai/gpt-oss-20b": 8000,
    "openai/gpt-oss-120b": 8000,
    "llama-3.1-8b-instant": 6000,
}
_FALLBACK_TPM_LIMIT = 6000  # para un modelo que no esté en el dict de arriba

_DEFAULT_SAFETY_MARGIN = 0.8
_WINDOW_SECONDS = 60


@dataclass
class _VentanaModelo:
    """Ventana deslizante de un solo modelo: su propio historial de uso."""
    tpm_limit: int
    safety_margin: float = _DEFAULT_SAFETY_MARGIN
    window_seconds: int = _WINDOW_SECONDS
    _historial: deque = field(default_factory=deque, repr=False)  # [(timestamp, tokens), ...]

    @property
    def presupuesto_efectivo(self) -> int:
        return int(self.tpm_limit * self.safety_margin)

    def _limpiar_historial(self) -> None:
        ahora = time.monotonic()
        while self._historial and (ahora - self._historial[0][0]) > self.window_seconds:
            self._historial.popleft()

    def tokens_usados(self) -> int:
        self._limpiar_historial()
        return sum(tokens for _, tokens in self._historial)

    def puede_proceder(self, tokens_estimados: int) -> bool:
        return (self.tokens_usados() + tokens_estimados) <= self.presupuesto_efectivo

    def tiempo_de_espera(self) -> float:
        self._limpiar_historial()
        if not self._historial:
            return 0.0
        entrada_mas_vieja_timestamp = self._historial[0][0]
        segundos_transcurridos = time.monotonic() - entrada_mas_vieja_timestamp
        return max(0.0, self.window_seconds - segundos_transcurridos)

    def registrar_uso(self, tokens: int) -> None:
        self._historial.append((time.monotonic(), tokens))
        self._limpiar_historial()


@dataclass
class RateLimiter:
    """
    Rate limiter multi-modelo: lleva una ventana deslizante INDEPENDIENTE
    por cada modelo de la cadena de Groq, reflejando que cada uno tiene
    su propia cuota de tokens/minuto separada.

    Un RateLimiter por sesión de Streamlit alcanza para este proyecto
    (no es necesario compartirlo entre usuarios — cada sesión tiene su
    propio presupuesto, igual que cada sesión tiene su propia
    SessionMemory en core/memory.py).
    """
    model_limits: dict[str, int] = field(default_factory=lambda: dict(MODEL_TPM_LIMITS))
    safety_margin: float = _DEFAULT_SAFETY_MARGIN
    window_seconds: int = _WINDOW_SECONDS
    _ventanas: dict[str, _VentanaModelo] = field(default_factory=dict, repr=False)

    def _ventana(self, model: str) -> _VentanaModelo:
        """Obtiene (o crea) la ventana deslizante de un modelo puntual."""
        if model not in self._ventanas:
            tpm = self.model_limits.get(model, _FALLBACK_TPM_LIMIT)
            self._ventanas[model] = _VentanaModelo(
                tpm_limit=tpm,
                safety_margin=self.safety_margin,
                window_seconds=self.window_seconds,
            )
        return self._ventanas[model]

    def presupuesto_efectivo(self, model: str) -> int:
        """Tokens/min disponibles para ESE modelo puntual, tras el margen de seguridad."""
        return self._ventana(model).presupuesto_efectivo

    def presupuesto_total_combinado(self) -> int:
        """Suma de los presupuestos efectivos de todos los modelos conocidos (~27.200 con la cadena actual)."""
        return sum(
            int(limite * self.safety_margin) for limite in self.model_limits.values()
        )

    def tokens_usados_en_ventana(self, model: str) -> int:
        return self._ventana(model).tokens_usados()

    def puede_proceder(self, model: str, tokens_estimados: int) -> bool:
        """¿Esta llamada a ESTE modelo entra en su presupuesto individual?"""
        return self._ventana(model).puede_proceder(tokens_estimados)

    def tiempo_de_espera(self, model: str) -> float:
        """Segundos hasta que se libere lugar en el presupuesto de ESE modelo."""
        return self._ventana(model).tiempo_de_espera()

    def registrar_uso(self, model: str, tokens: int) -> None:
        """Registra tokens consumidos en ESE modelo puntual."""
        self._ventana(model).registrar_uso(tokens)

    def siguiente_modelo_disponible(
        self, cadena: list[str], tokens_estimados: int
    ) -> str | None:
        """
        Recorre la cadena de modelos en orden (mismo orden que
        GROQ_MODEL_CHAIN en core/llm_client.py) y devuelve el primero
        que todavía tiene presupuesto para esta llamada, o None si
        ninguno lo tiene.

        Uso típico antes de llamar a chat_completion(), para evitar
        pedirle a Groq un modelo que ya sabemos que va a rechazar por
        cuota:

            modelo = limiter.siguiente_modelo_disponible(GROQ_MODEL_CHAIN, tokens_estimados)
            if modelo is None:
                # esperar limiter.tiempo_de_espera(GROQ_MODEL_CHAIN[0]) antes de reintentar
        """
        for modelo in cadena:
            if self.puede_proceder(modelo, tokens_estimados):
                return modelo
        return None


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
    # Ejemplo rápido: simula la cadena completa de 4 modelos, mostrando
    # que cada uno tiene su propio presupuesto independiente.
    limiter = RateLimiter()
    cadena = list(MODEL_TPM_LIMITS.keys())

    print("Presupuesto por modelo (tras margen de seguridad del 80%):")
    for modelo in cadena:
        print(f"  {modelo:28s} -> {limiter.presupuesto_efectivo(modelo)} tokens/min")
    print(f"\nPresupuesto TOTAL combinado: {limiter.presupuesto_total_combinado()} tokens/min")

    mensajes_ejemplo = [
        {"role": "system", "content": "Sos un agente de calidad de datos." * 20},
        {"role": "user", "content": "Rellename los nulos de edad con la media"},
    ]
    tokens_estimados = estimar_tokens(mensajes_ejemplo)
    print(f"\nTokens estimados para un mensaje típico: {tokens_estimados}")

    print("\n--- Simulamos llamadas seguidas hasta agotar el modelo 1, y ver que salta al modelo 2 ---")
    for intento in range(1, 90):
        modelo = limiter.siguiente_modelo_disponible(cadena, tokens_estimados)
        if modelo is None:
            print(f"Llamada {intento}: TODA la cadena sin presupuesto — esperar antes de reintentar.")
            break
        limiter.registrar_uso(modelo, tokens_estimados)
        if intento % 10 == 0 or intento <= 3:
            print(f"Llamada {intento}: usó '{modelo}' "
                  f"(usado en ese modelo: {limiter.tokens_usados_en_ventana(modelo)})")