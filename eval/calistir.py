"""Measure retrieval quality, and compare embedding models.

Retrieval either finds the right passage or it does not, and that is
measurable without involving the LLM at all. Separating the two matters:
if a wrong answer comes out of the app, this script tells you whether the
fault lies in retrieval or in the prompt.

Metrics
-------
recall@k  Share of questions whose correct page appears in the top k.
          Answers "did we give the LLM a chance?"
MRR       Mean reciprocal rank: 1/rank of the first correct hit,
          averaged. Rewards ranking the right passage first, not third.

Usage
-----
    python -m eval.calistir
    python -m eval.calistir --models intfloat/multilingual-e5-base \
                                     sentence-transformers/all-MiniLM-L6-v2
"""

import argparse
import json
from pathlib import Path

import chromadb

from rag import config
from rag.chunker import chunk_pages
from rag.embedder import Embedder
from rag.loader import load_pdf
from rag.store import ingest, search

QUESTIONS_FILE = Path(__file__).parent / "sorular.json"


def load_questions() -> list[dict]:
    """Read the evaluation set."""
    with open(QUESTIONS_FILE, encoding="utf-8") as handle:
        return json.load(handle)


def build_temp_collection(model_name: str, pdf_path: Path, embedder: Embedder):
    """Index the PDF into a throwaway in-memory collection.

    Each model produces vectors of a different dimension and meaning, so
    they cannot share a collection. An in-memory client keeps the real
    chroma_db/ directory untouched by experiments.
    """
    client = chromadb.EphemeralClient()
    safe_name = model_name.replace("/", "_")[:60]

    try:
        collection = client.create_collection(
            name=safe_name, configuration={"hnsw": {"space": "cosine"}}
        )
    except TypeError:
        collection = client.create_collection(
            name=safe_name, metadata={"hnsw:space": "cosine"}
        )

    pages = load_pdf(pdf_path)
    chunks = chunk_pages(pages, source=pdf_path.name)
    ingest(collection, chunks, embedder)
    return collection, len(chunks)


def evaluate(collection, embedder: Embedder, questions: list[dict], k: int) -> dict:
    """Run every question and score the ranking."""
    hit_count = 0
    reciprocal_ranks = []
    details = []

    for item in questions:
        expected_pages = set(item["expected_pages"])
        hits = search(collection, item["question"], embedder, top_k=k)

        rank = None
        for position, hit in enumerate(hits, start=1):
            if hit["page"] in expected_pages:
                rank = position
                break

        if rank is not None:
            hit_count += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)

        details.append(
            {
                "question": item["question"],
                "rank": rank,
                "top_distance": hits[0]["distance"] if hits else None,
                "returned_pages": [h["page"] for h in hits],
            }
        )

    total = len(questions)
    return {
        "recall": hit_count / total,
        "mrr": sum(reciprocal_ranks) / total,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="PDF to evaluate against. Defaults to the only file in data/.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[config.EMBEDDING_MODEL],
        help="One or more sentence-transformers model names to compare.",
    )
    parser.add_argument("--k", type=int, default=3, help="Cut-off for recall@k.")
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-question results."
    )
    args = parser.parse_args()

    pdf_path = args.pdf
    if pdf_path is None:
        candidates = sorted(config.DATA_DIR.glob("*.pdf"))
        if len(candidates) != 1:
            parser.error(
                f"Found {len(candidates)} PDFs in data/. Pass --pdf explicitly."
            )
        pdf_path = candidates[0]

    questions = load_questions()
    print(f"Document: {pdf_path.name}")
    print(f"Questions: {len(questions)}   Cut-off: k={args.k}\n")

    results = {}
    for model_name in args.models:
        print(f"Evaluating {model_name} ...")
        embedder = Embedder(model_name)
        collection, chunk_count = build_temp_collection(
            model_name, pdf_path, embedder
        )
        results[model_name] = evaluate(collection, embedder, questions, args.k)
        results[model_name]["chunks"] = chunk_count

    print(f"\n{'model':<50} {'recall@' + str(args.k):>10} {'MRR':>8}")
    print("-" * 70)
    for model_name, scores in results.items():
        print(
            f"{model_name:<50} {scores['recall']:>10.3f} {scores['mrr']:>8.3f}"
        )

    if args.verbose:
        for model_name, scores in results.items():
            print(f"\n--- {model_name} ---")
            for detail in scores["details"]:
                status = f"rank {detail['rank']}" if detail["rank"] else "MISS"
                print(f"  [{status:>7}] {detail['question']}")
                print(f"            returned pages: {detail['returned_pages']}")


if __name__ == "__main__":
    main()
