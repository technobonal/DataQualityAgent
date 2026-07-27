"""
rag/validator.py

Dos validaciones para el pipeline de RAG:

  1. Validar el diccionario de datos ANTES de indexarlo (rag/ingest.py
     puede extraer chunks de un archivo vacío, corrupto, o con muy poco
     contenido útil — sin esta validación, se guardaría "basura" en el
     vector store sin que nadie se entere).

  2. Validar la RELEVANCIA de lo que Chroma recupera en una búsqueda
     (rag/vectorstore.py) antes de dárselo al LLM como contexto — un
     chunk con una distancia muy alta (poco parecido semánticamente a la
     pregunta) puede confundir más de lo que ayuda si se lo pasamos igual.

Sin dependencia de LLM — ambas validaciones son deterministas.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from rag.ingest import DocumentChunk
from rag.vectorstore import RetrievedChunk

# Umbral de distancia de Chroma por debajo del cual un chunk se considera
# "relevante". Con embeddings de Cohere y distancia coseno, valores por
# encima de esto suelen ser resultados poco relacionados con la pregunta
# (más una coincidencia forzada que una respuesta real). Es un valor de
# partida razonable — conviene ajustarlo con casos reales del proyecto.
_UMBRAL_DISTANCIA_RELEVANTE = 0.8

# Cantidad mínima de chunks "sustanciales" (no vacíos ni triviales) para
# considerar que el diccionario de datos aporta algo útil.
_MIN_CHUNKS_UTILES = 1

# Un chunk con menos caracteres que esto se considera "trivial" (ej. un
# encabezado suelto o una celda vacía que igual generó texto).
_MIN_CARACTERES_POR_CHUNK = 5


@dataclass
class DocumentValidationResult:
    """Resultado de validar un diccionario de datos antes de indexarlo."""
    is_valid: bool
    total_chunks: int
    useful_chunks: int
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validar_documento(chunks: list[DocumentChunk]) -> DocumentValidationResult:
    """
    Valida los chunks que devolvió rag/ingest.py ANTES de guardarlos en
    el vector store. No modifica nada — solo diagnostica.

    Rechaza (is_valid=False) cuando:
      - No se generó ningún chunk (archivo vacío o no se pudo extraer nada).
      - Todos los chunks son triviales (muy cortos, sin contenido real).
    """
    issues: list[str] = []

    if not chunks:
        return DocumentValidationResult(
            is_valid=False,
            total_chunks=0,
            useful_chunks=0,
            issues=["El documento no generó ningún chunk — puede estar vacío o no se pudo extraer su contenido."],
        )

    chunks_utiles = [c for c in chunks if len(c.text.strip()) >= _MIN_CARACTERES_POR_CHUNK]
    chunks_triviales = len(chunks) - len(chunks_utiles)

    if chunks_triviales > 0:
        issues.append(f"{chunks_triviales} chunk(s) son demasiado cortos para aportar contexto útil.")

    if len(chunks_utiles) < _MIN_CHUNKS_UTILES:
        issues.append("El documento no tiene suficiente contenido útil para ser indexado.")
        return DocumentValidationResult(
            is_valid=False,
            total_chunks=len(chunks),
            useful_chunks=len(chunks_utiles),
            issues=issues,
        )

    return DocumentValidationResult(
        is_valid=True,
        total_chunks=len(chunks),
        useful_chunks=len(chunks_utiles),
        issues=issues,
    )


@dataclass
class RetrievalValidationResult:
    """Resultado de filtrar chunks recuperados por relevancia real."""
    relevant_chunks: list[dict[str, Any]] = field(default_factory=list)
    discarded_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def filtrar_por_relevancia(
    chunks: list[RetrievedChunk],
    umbral_distancia: float = _UMBRAL_DISTANCIA_RELEVANTE,
) -> RetrievalValidationResult:
    """
    Filtra los chunks recuperados por rag/vectorstore.py, descartando los
    que tengan una distancia semántica demasiado alta (poco relacionados
    con la pregunta) — evita pasarle al LLM contexto irrelevante que
    podría confundirlo más de lo que ayuda.

    Si TODOS los chunks quedan descartados, devuelve la lista vacía — es
    una señal correcta de "el diccionario de datos no tiene información
    relevante para esta pregunta", no un error.
    """
    relevantes = [c for c in chunks if c.distance <= umbral_distancia]
    descartados = len(chunks) - len(relevantes)

    return RetrievalValidationResult(
        relevant_chunks=[c.to_dict() for c in relevantes],
        discarded_count=descartados,
    )


if __name__ == "__main__":
    from rag.ingest import DocumentChunk
    from rag.vectorstore import RetrievedChunk

    print("--- 1. Documento válido ---")
    chunks_ok = [
        DocumentChunk(text="edad: Edad del cliente en años, 0 a 120.", source_file="dic.csv", chunk_type="structured_row"),
        DocumentChunk(text="estado: activo, inactivo o pendiente.", source_file="dic.csv", chunk_type="structured_row"),
    ]
    print(validar_documento(chunks_ok).to_dict())

    print("\n--- 2. Documento vacío (sin chunks) ---")
    print(validar_documento([]).to_dict())

    print("\n--- 3. Documento con chunks triviales ---")
    chunks_triviales = [
        DocumentChunk(text="", source_file="dic.csv", chunk_type="free_text"),
        DocumentChunk(text="  ", source_file="dic.csv", chunk_type="free_text"),
    ]
    print(validar_documento(chunks_triviales).to_dict())

    print("\n--- 4. Filtrar por relevancia (mezcla de relevantes e irrelevantes) ---")
    resultados_busqueda = [
        RetrievedChunk(text="edad: 0 a 120", source_file="dic.csv", distance=0.35),
        RetrievedChunk(text="estado: activo/inactivo", source_file="dic.csv", distance=0.50),
        RetrievedChunk(text="algo sin relación real", source_file="dic.csv", distance=1.2),
    ]
    print(filtrar_por_relevancia(resultados_busqueda).to_dict())

    print("\n--- 5. Filtrar por relevancia (ninguno relevante) ---")
    todos_irrelevantes = [RetrievedChunk(text="ruido", source_file="dic.csv", distance=1.5)]
    print(filtrar_por_relevancia(todos_irrelevantes).to_dict())