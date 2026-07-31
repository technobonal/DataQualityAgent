"""
security/rate_limiter.py

Limitador de tasa (rate limiter) del lado del cliente: evita que el
agente le pegue a Groq con más tokens/requests por minuto de los que
cada modelo de la cadena permite, en vez de descubrirlo recién cuando
Groq devuelve un 429 (como pasó durante las pruebas del proyecto — ver
core/llm_client.py).

REDISEÑO (v2 -- agrega tracking de RPM): cada modelo de GROQ_MODEL_CHAIN
tiene su PROPIA cuota de tokens/minuto Y de requests/minuto en Groq (no
un pool único compartido):

    llama-3.3-70b-versatile   -> 12.000 TPM / 30 RPM
    openai/gpt-oss-20b        ->  8.000 TPM / 30 RPM
    openai/gpt-oss-120b       ->  8.000 TPM / 30 RPM
    llama-3.1-8b-instant      ->  6.000 TPM / 30 RPM

MOTIVO DEL AGREGADO DE RPM: la v1 de este archivo solo trackeaba TPM.
En la práctica, con varias tool-calls seguidas en poco tiempo (loop de
_MAX_TOOL_ITERATIONS en core/agent.py), Groq corta antes por RPM que
por TPM -- el rate limiter decía "hay cupo" (porque medía tokens) y
Groq devolvía 429 igual (porque el límite real que se agotó fue el de
requests/minuto). Groq corta por el eje que se agote primero, así que
ahora nosotros chequeamos los dos.

Esto NO es sobre autenticación/autorización — es sobre protección de
cuota: una sesión no debería poder "quemar" sin querer los tokens/min o
requests/min disponibles de un modelo pidiendo demasiadas cosas seguidas.

Sin dependencia de un LLM en sí — solo lleva la cuenta de tokens y
requests usados en una ventana de tiempo deslizante. litellm se usa
únicamente para ESTIMAR cuántos tokens va a pesar una llamada antes de
hacerla (no para llamar al modelo).
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

# NUEVO: cuota RPM real por modelo (plan free de Groq: ~30 requests/min
# para casi todos los modelos). Este es el eje que faltaba trackear --
# ver la nota de "MOTIVO DEL AGREGADO" más arriba.
MODEL_RPM_LIMITS: dict[str, int] = {
    "llama-3.3-70b-versatile": 30,
    "openai/gpt-oss-20b": 30,
    "openai/gpt-oss-120b": 30,
    "llama-3.1-8b-instant": 30,
}
_FALLBACK_RPM_LIMIT = 30  # para un modelo que no esté en el dict de arriba

_DEFAULT_SAFETY_MARGIN = 0.8
_WINDOW_SECONDS = 60


@dataclass
class _VentanaModelo:
    """Ventana deslizante de un solo modelo: su propio historial de uso,
    tanto de tokens (TPM) como de requests (RPM)."""
    tpm_limit: int
    rpm_limit: int = _FALLBACK_RPM_LIMIT
    safety_margin: float = _DEFAULT_SAFETY_MARGIN
    window_seconds: int = _WINDOW_SECONDS
    _historial: deque = field(default_factory=deque, repr=False)  # [(timestamp, tokens), ...]
    _historial_requests: deque = field(default_factory=deque, repr=False)  # [timestamp, ...]

    @property
    def presupuesto_efectivo(self) -> int:
        return int(self.tpm_limit * self.safety_margin)

    @property
    def rpm_efectivo(self) -> int:
        """Requests/min disponibles tras el margen de seguridad (mínimo 1,
        para no dejar un modelo inutilizable si el margen lo redondea a 0)."""
        return max(1, int(self.rpm_limit * self.safety_margin))

    def _limpiar_historial(self) -> None:
        ahora = time.monotonic()
        while self._historial and (ahora - self._historial[0][0]) > self.window_seconds:
            self._historial.popleft()

    def _limpiar_historial_requests(self) -> None:
        ahora = time.monotonic()
        while self._historial_requests and (ahora - self._historial_requests[0]) > self.window_seconds:
            self._historial_requests.popleft()

    def tokens_usados(self) -> int:
        self._limpiar_historial()
        return sum(tokens for _, tokens in self._historial)

    def requests_usados(self) -> int:
        """Cuántos requests (éxitos o fallidos) se hicieron a este modelo
        en la ventana actual de 60s."""
        self._limpiar_historial_requests()
        return len(self._historial_requests)

    def puede_proceder(self, tokens_estimados: int) -> bool:
        """Chequea AMBOS ejes: si cualquiera de los dos (TPM o RPM) está
        al límite, no se puede proceder -- Groq corta por el que se
        agote primero, así que nosotros hacemos lo mismo."""
        tpm_ok = (self.tokens_usados() + tokens_estimados) <= self.presupuesto_efectivo
        rpm_ok = self.requests_usados() < self.rpm_efectivo
        return tpm_ok and rpm_ok

    def tiempo_de_espera(self) -> float:
        """Segundos hasta que se libere cupo, considerando el eje (TPM o
        RPM) que tarde más en liberarse."""
        self._limpiar_historial()
        self._limpiar_historial_requests()
        esperas = []
        if self._historial:
            esperas.append(max(0.0, self.window_seconds - (time.monotonic() - self._historial[0][0])))
        if self._historial_requests:
            esperas.append(max(0.0, self.window_seconds - (time.monotonic() - self._historial_requests[0])))
        return max(esperas) if esperas else 0.0

    def registrar_uso(self, tokens: int) -> None:
        """Registra tokens consumidos -- llamar SOLO cuando la llamada
        tuvo éxito (los tokens realmente se procesaron)."""
        self._historial.append((time.monotonic(), tokens))
        self._limpiar_historial()

    def registrar_request(self) -> None:
        """Registra UN intento de llamada a Groq -- llamar SIEMPRE que se
        intenta la llamada, haya tenido éxito o no (el cupo de RPM se
        gasta igual aunque la respuesta sea un 429)."""
        self._historial_requests.append(time.monotonic())
        self._limpiar_historial_requests()


@dataclass
class RateLimiter:
    """
    Rate limiter multi-modelo: lleva una ventana deslizante INDEPENDIENTE
    por cada modelo de la cadena de Groq, reflejando que cada uno tiene
    su propia cuota de tokens/minuto Y requests/minuto separada.

    Un RateLimiter por sesión de Streamlit alcanza para este proyecto
    (no es necesario compartirlo entre usuarios — cada sesión tiene su
    propio presupuesto, igual que cada sesión tiene su propia
    SessionMemory en core/memory.py).
    """
    model_limits: dict[str, int] = field(default_factory=lambda: dict(MODEL_TPM_LIMITS))
    model_rpm_limits: dict[str, int] = field(default_factory=lambda: dict(MODEL_RPM_LIMITS))
    safety_margin: float = _DEFAULT_SAFETY_MARGIN
    window_seconds: int = _WINDOW_SECONDS
    _ventanas: dict[str, _VentanaModelo] = field(default_factory=dict, repr=False)

    def _ventana(self, model: str) -> _VentanaModelo:
        """Obtiene (o crea) la ventana deslizante de un modelo puntual."""
        if model not in self._ventanas:
            tpm = self.model_limits.get(model, _FALLBACK_TPM_LIMIT)
            rpm = self.model_rpm_limits.get(model, _FALLBACK_RPM_LIMIT)
            self._ventanas[model] = _VentanaModelo(
                tpm_limit=tpm,
                rpm_limit=rpm,
                safety_margin=self.safety_margin,
                window_seconds=self.window_seconds,
            )
        return self._ventanas[model]

    def presupuesto_efectivo(self, model: str) -> int:
        """Tokens/min disponibles para ESE modelo puntual, tras el margen de seguridad."""
        return self._ventana(model).presupuesto_efectivo

    def rpm_efectivo(self, model: str) -> int:
        """Requests/min disponibles para ESE modelo puntual, tras el margen de seguridad."""
        return self._ventana(model).rpm_efectivo

    def presupuesto_total_combinado(self) -> int:
        """Suma de los presupuestos efectivos de todos los modelos conocidos (~27.200 con la cadena actual)."""
        return sum(
            int(limite * self.safety_margin) for limite in self.model_limits.values()
        )

    def tokens_usados_en_ventana(self, model: str) -> int:
        return self._ventana(model).tokens_usados()

    def requests_usados_en_ventana(self, model: str) -> int:
        """Cuántos requests se hicieron a ESE modelo en la ventana actual."""
        return self._ventana(model).requests_usados()

    def puede_proceder(self, model: str, tokens_estimados: int) -> bool:
        """¿Esta llamada a ESTE modelo entra en su presupuesto individual
        (TPM y RPM a la vez)?"""
        return self._ventana(model).puede_proceder(tokens_estimados)

    def tiempo_de_espera(self, model: str) -> float:
        """Segundos hasta que se libere lugar en el presupuesto de ESE modelo."""
        return self._ventana(model).tiempo_de_espera()

    def registrar_uso(self, model: str, tokens: int) -> None:
        """Registra tokens consumidos en ESE modelo puntual (solo en éxito)."""
        self._ventana(model).registrar_uso(tokens)

    def registrar_request(self, model: str) -> None:
        """Registra un intento de llamada a ESTE modelo (éxito o fallo).
        Llamar ANTES de cada intento real a Groq, para que el cupo de
        RPM refleje intentos reales aunque terminen en 429."""
        self._ventana(model).registrar_request()

    def siguiente_modelo_disponible(
        self, cadena: list[str], tokens_estimados: int
    ) -> str | None:
        """
        Recorre la cadena de modelos en orden (mismo orden que
        GROQ_MODEL_CHAIN en core/llm_client.py) y devuelve el primero
        que todavía tiene presupuesto (TPM y RPM) para esta llamada, o
        None si ninguno lo tiene.

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
    # que cada uno tiene su propio presupuesto independiente de TPM Y
    # de RPM, y que el RPM puede agotarse ANTES que el TPM.
    limiter = RateLimiter()
    cadena = list(MODEL_TPM_LIMITS.keys())

    print("Presupuesto por modelo (tras margen de seguridad del 80%):")
    for modelo in cadena:
        print(
            f"  {modelo:28s} -> {limiter.presupuesto_efectivo(modelo)} tokens/min "
            f"| {limiter.rpm_efectivo(modelo)} requests/min"
        )
    print(f"\nPresupuesto TOTAL combinado (TPM): {limiter.presupuesto_total_combinado()} tokens/min")

    mensajes_ejemplo = [
        {"role": "system", "content": "Sos un agente de calidad de datos." * 20},
        {"role": "user", "content": "Rellename los nulos de edad con la media"},
    ]
    tokens_estimados = estimar_tokens(mensajes_ejemplo)
    print(f"\nTokens estimados para un mensaje típico: {tokens_estimados}")

    print("\n--- Simulamos llamadas seguidas: con un mensaje chico, el RPM se agota ANTES que el TPM ---")
    modelo_1 = cadena[0]
    for intento in range(1, 40):
        modelo = limiter.siguiente_modelo_disponible(cadena, tokens_estimados)
        if modelo is None:
            print(f"Llamada {intento}: TODA la cadena sin presupuesto (TPM o RPM) — esperar antes de reintentar.")
            break
        limiter.registrar_request(modelo)
        limiter.registrar_uso(modelo, tokens_estimados)
        if intento % 5 == 0 or intento <= 3:
            print(
                f"Llamada {intento}: usó '{modelo}' "
                f"(tokens: {limiter.tokens_usados_en_ventana(modelo)}, "
                f"requests: {limiter.requests_usados_en_ventana(modelo)}/{limiter.rpm_efectivo(modelo)})"
            )