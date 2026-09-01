"""Wrap a local sentence-transformers model.

Two reasons this is a class rather than a pair of functions: the model
weights are expensive to load and must be reused, and the query/passage
asymmetry is easy to get wrong if the prefix logic is scattered across
call sites.
"""

from sentence_transformers import SentenceTransformer

from rag import config


class Embedder:
    """Encode text into vectors with the correct e5 prefixes applied."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL) -> None:
        """Load the model. Downloads it on first use, then reads cache."""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        # Models outside the e5 family do not use prefixes at all.
        self.uses_prefix = "e5" in model_name.lower()

    @property
    def dimension(self) -> int:
        """Vector length produced by this model."""
        # Renamed in newer sentence-transformers; support both.
        if hasattr(self.model, "get_embedding_dimension"):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()

    def _encode(self, texts: list[str], prefix: str) -> list[list[float]]:
        """Encode a batch, normalising vectors to unit length.

        Normalising matters: with unit vectors, cosine similarity reduces
        to a dot product, and Chroma's cosine distance behaves predictably.
        """
        if self.uses_prefix:
            texts = [prefix + t for t in texts]

        vectors = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Encode document chunks for storage."""
        if not texts:
            return []
        return self._encode(texts, config.PASSAGE_PREFIX)

    def embed_query(self, text: str) -> list[float]:
        """Encode a single user question for search."""
        return self._encode([text], config.QUERY_PREFIX)[0]


if __name__ == "__main__":
    # Two checks in one run.
    #
    # First, that the prefixes do their job: a question and its answer are
    # worded differently, and the asymmetric encoding is what bridges them.
    # The relevant passage should score higher than the distractor.
    #
    # Second, that the model is genuinely multilingual. The same pair is
    # repeated in another language, and the ordering should hold there
    # too -- which is the entire reason for choosing a 1.1 GB model over
    # a 90 MB English one.
    embedder = Embedder()
    print(f"Model: {embedder.model_name}  Dimension: {embedder.dimension}")

    cases = [
        (
            "English",
            "How do I clean the device?",
            [
                "To clean the device, unplug it first and wipe it with a damp cloth.",
                "The warranty runs for two years from the date of purchase.",
            ],
        ),
        (
            "Turkish",
            "Cihazı nasıl temizlerim?",
            [
                "Temizlik için cihazı fişten çekin ve nemli bezle silin.",
                "Garanti süresi satın alma tarihinden itibaren iki yıldır.",
            ],
        ),
    ]

    for language, question, passages in cases:
        print(f"\n{language}: {question}")
        query_vector = embedder.embed_query(question)
        passage_vectors = embedder.embed_passages(passages)
        for passage, vector in zip(passages, passage_vectors):
            similarity = sum(a * b for a, b in zip(query_vector, vector))
            print(f"  {similarity:.3f}  {passage[:55]}")