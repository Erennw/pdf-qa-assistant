"""Vector store: persist chunks in ChromaDB and search them.

Embeddings are computed by our own Embedder and handed to Chroma
explicitly. Chroma can generate embeddings itself, but then the model
choice hides inside the database and cannot be compared in an experiment.
"""

import chromadb

from rag import config
from rag.embedder import Embedder


def get_client() -> chromadb.ClientAPI:
    """Open the on-disk Chroma database."""
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def get_collection(client: chromadb.ClientAPI):
    """Fetch or create the collection, configured for cosine distance.

    Chroma defaults to squared L2. With normalised vectors the ranking
    comes out the same, but the numbers are not cosine distances, so any
    threshold tuned on them would be meaningless. Setting the space
    explicitly keeps MAX_DISTANCE interpretable.

    The configuration argument replaced the old metadata form in recent
    Chroma versions; the fallback keeps older installs working.
    """
    try:
        return client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            configuration={"hnsw": {"space": "cosine"}},
        )
    except TypeError:
        return client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )


def ingest(collection, chunks: list[dict], embedder: Embedder) -> dict:
    """Add chunks to the collection, skipping ones already stored.

    Deterministic IDs make this idempotent: running it twice on the same
    document embeds nothing the second time. Embedding is the slow part
    of the pipeline, so the check pays for itself immediately.

    Returns:
        {"added": int, "skipped": int} for reporting in the UI.
    """
    if not chunks:
        return {"added": 0, "skipped": 0}

    incoming_ids = [c["id"] for c in chunks]
    existing = collection.get(ids=incoming_ids, include=[])
    existing_ids = set(existing["ids"])

    new_chunks = [c for c in chunks if c["id"] not in existing_ids]
    if not new_chunks:
        return {"added": 0, "skipped": len(chunks)}

    texts = [c["text"] for c in new_chunks]
    vectors = embedder.embed_passages(texts)

    collection.add(
        ids=[c["id"] for c in new_chunks],
        documents=texts,
        embeddings=vectors,
        metadatas=[
            {
                "source": c["source"],
                "page": c["page"],
                "label": c["label"],
                "position": c["position"],
            }
            for c in new_chunks
        ],
    )

    return {"added": len(new_chunks), "skipped": len(chunks) - len(new_chunks)}


def search(
    collection,
    question: str,
    embedder: Embedder,
    top_k: int = config.TOP_K,
    source: str | None = None,
) -> list[dict]:
    """Return the chunks most similar to the question.

    Args:
        source: Restrict the search to one document. Without it, a
            question is answered from every PDF in the collection, which
            is rarely what the user means when one file is selected.

    Returns:
        A list of {"text", "page", "label", "source", "distance"} ordered by
        distance, closest first. Empty if the collection has no data.
    """
    if collection.count() == 0:
        return []

    query_vector = embedder.embed_query(question)
    where = {"source": source} if source else None

    response = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for text, metadata, distance in zip(
        response["documents"][0],
        response["metadatas"][0],
        response["distances"][0],
    ):
        hits.append(
            {
                "text": text,
                "page": metadata["page"],
                # Older records predate page labels; fall back safely.
                "label": metadata.get("label", str(metadata["page"])),
                "source": metadata["source"],
                "distance": distance,
            }
        )
    return hits


def list_sources(collection) -> list[str]:
    """Names of every document currently indexed."""
    if collection.count() == 0:
        return []
    records = collection.get(include=["metadatas"])
    return sorted({m["source"] for m in records["metadatas"]})


def delete_source(collection, source: str) -> int:
    """Remove every chunk belonging to one document.

    Returns the number of chunks deleted.
    """
    records = collection.get(where={"source": source}, include=[])
    ids = records["ids"]
    if ids:
        collection.delete(ids=ids)
    return len(ids)
