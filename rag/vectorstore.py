"""
rag/vectorstore.py

Vector store del RAG: guarda los chunks del diccionario de datos (ver
rag/ingest.py) en Chroma, embebidos con Cohere (embed-multilingual-v3.0
por defecto — el mismo modelo que ya se usó en el proyecto LogisCore),
y los recupera por similitud semántica cuando el agente necesita
contexto sobre una columna.

Detalle importante sobre Cohere v3: los modelos v3 necesitan un
input_type distinto según el caso — "search_document" para lo que se
GUARDA, "search_query" para lo que se BUSCA. Usar el mismo input_type
para ambos casos no rompe nada, pero degrada la calidad de la búsqueda.
La integración automática de Chroma para Cohere siempre usa
"search_document", incluso al buscar — por eso acá NO se usa esa
integración para las queries: se embebe la pregunta a mano, con el
input_type correcto, y se le pasa a Chroma ya calculada.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

from rag.ingest import DocumentChunk

load_dotenv()

_COHERE_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-multilingual-v3.0")
_CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
_DEFAULT_COLLECTION = "diccionario_datos"


def _get_cohere_api_key() -> str:
    api_key = os.getenv("COHERE_API_KEY", "")
    if not api_key:
        raise ValueError(
            "Falta COHERE_API_KEY en el archivo .env. Conseguila en "
            "https://dashboard.cohere.com/api-keys"
        )
    return api_key


def _get_chroma_client() -> chromadb.ClientAPI:
    """Cliente persistente de Chroma — guarda en disco, en CHROMA_DB_PATH."""
    return chromadb.PersistentClient(path=_CHROMA_DB_PATH)


def _get_embedding_function() -> embedding_functions.CohereEmbeddingFunction:
    """
    Embedding function para INDEXAR documentos (usa input_type=
    'search_document' internamente — correcto para lo que se guarda).
    Para embeber QUERIES se usa _embed_query() por separado, con
    input_type='search_query'; ver nota del módulo.
    """
    return embedding_functions.CohereEmbeddingFunction(
        api_key=_get_cohere_api_key(),
        model_name=_COHERE_MODEL,
    )


def _embed_query(texto: str) -> list[float]:
    """Embebe una QUERY (pregunta de búsqueda) con el input_type correcto para Cohere v3."""
    import cohere
    cliente = cohere.Client(_get_cohere_api_key())
    respuesta = cliente.embed(texts=[texto], model=_COHERE_MODEL, input_type="search_query")
    return list(respuesta.embeddings[0])


def get_or_create_collection(collection_name: str = _DEFAULT_COLLECTION):
    """Obtiene (o crea si no existe) la colección de Chroma para el diccionario de datos."""
    cliente = _get_chroma_client()
    return cliente.get_or_create_collection(
        name=collection_name,
        embedding_function=_get_embedding_function(),
    )


@dataclass
class IngestSummary:
    """Resumen de cuántos chunks se guardaron en el vector store."""
    collection_name: str
    chunks_added: int
    source_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def store_chunks(
    chunks: list[DocumentChunk],
    collection_name: str = _DEFAULT_COLLECTION,
) -> IngestSummary:
    """
    Embebe y guarda una lista de chunks (salida de rag/ingest.py) en Chroma.

    Cada chunk se guarda con un id único (archivo de origen + posición),
    y su metadata (source_file, chunk_type, etc.) queda disponible para
    filtrar búsquedas después. Usa upsert (no insert): si se vuelve a
    ingestar el mismo archivo, actualiza en vez de duplicar.
    """
    if not chunks:
        return IngestSummary(collection_name=collection_name, chunks_added=0)

    coleccion = get_or_create_collection(collection_name)

    ids = [f"{c.source_file}::{i}" for i, c in enumerate(chunks)]
    documentos = [c.text for c in chunks]
    metadatas = [
        {
            "source_file": c.source_file,
            "chunk_type": c.chunk_type,
            **{k: str(v) for k, v in c.metadata.items()},
        }
        for c in chunks
    ]

    coleccion.upsert(ids=ids, documents=documentos, metadatas=metadatas)

    return IngestSummary(
        collection_name=collection_name,
        chunks_added=len(chunks),
        source_files=sorted({c.source_file for c in chunks}),
    )


@dataclass
class RetrievedChunk:
    """Un chunk recuperado de la búsqueda semántica, con su distancia (menor = más relevante)."""
    text: str
    source_file: str
    distance: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def query_dictionary(
    pregunta: str,
    n_results: int = 3,
    collection_name: str = _DEFAULT_COLLECTION,
) -> list[RetrievedChunk]:
    """
    Busca en el diccionario de datos los chunks más relevantes para una
    pregunta (ej. "¿qué significa la columna edad?").

    Devuelve lista vacía si la colección todavía no tiene nada indexado
    (en vez de lanzar un error) — típico al arrancar sin haber subido
    todavía un diccionario de datos.
    """
    coleccion = get_or_create_collection(collection_name)

    if coleccion.count() == 0:
        return []

    query_embedding = _embed_query(pregunta)
    resultados = coleccion.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, coleccion.count()),
    )

    documentos = resultados.get("documents", [[]])[0]
    metadatas = resultados.get("metadatas", [[]])[0]
    distancias = resultados.get("distances", [[]])[0]

    return [
        RetrievedChunk(
            text=texto,
            source_file=metadata.get("source_file", "desconocido"),
            distance=round(float(distancia), 4),
            metadata=metadata,
        )
        for texto, metadata, distancia in zip(documentos, metadatas, distancias)
    ]


def limpiar_coleccion(collection_name: str = _DEFAULT_COLLECTION) -> None:
    """Borra toda la colección — útil para volver a cargar el diccionario de datos desde cero."""
    cliente = _get_chroma_client()
    try:
        cliente.delete_collection(collection_name)
    except Exception:
        pass  # si no existía, no hay nada que borrar


if __name__ == "__main__":
    # Prueba manual — requiere COHERE_API_KEY real en el .env.
    # IMPORTANTE: no se puede correr desde el sandbox de Claude (sin
    # acceso a api.cohere.com) — la lógica de guardado/búsqueda en Chroma
    # ya se probó con embeddings simulados (ver conversación). Correr
    # esto desde tu máquina para confirmar contra Cohere real.
    from rag.ingest import ingest_document

    chunks = ingest_document("diccionario_de_datos.csv")
    resumen = store_chunks(chunks)
    print(resumen.to_dict())

    resultados = query_dictionary("¿qué significa la columna edad?")
    for r in resultados:
        print(r.to_dict())