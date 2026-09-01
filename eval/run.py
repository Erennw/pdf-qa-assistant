"""Measure retrieval quality, and compare embedding models.

Retrieval either finds the right passage or it does not, and that is
measurable without involving the LLM at all. Separating the two matters:
if a wrong answer comes out of the app, this script tells you whether the
fault lies in retrieval or in the prompt.

Metrics
-------
recall@k  Share of answerable questions whose correct page appears in the
          top k. Answers "did we give the LLM a chance?"
MRR       Mean reciprocal rank: 1/rank of the first correct hit,
          averaged. Rewards ranking the right passage first, not third.

Control questions
-----------------
Items with an empty expected_pages list are questions the document does
not answer. They are excluded from recall and MRR -- a question with no
correct page would score as a miss and quietly depress both numbers --
and reported separately. The distance of their best hit is what sets
MAX_DISTANCE, so this script prints both bands and a suggested value.

Usage
-----
    python -m eval.run --pdf data/doc.pdf --questions eval/questions_en.json
    python -m eval.run --models intfloat/multilingual-e5-base \
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

EVAL_DIR = Path(__file__).parent
DEFAULT_QUESTIONS = EVAL_DIR / "questions_en.json"


def load_questions(path: Path) -> list[dict]:
    """Read an evaluation set.

    Each language gets its own file. Pooling them into one average would
    hide the very difference this evaluation exists to measure.
    """
    with open(path, encoding="utf-8") as handle:
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
    """Score ranking on answerable questions; profile control questions."""
    answerable = [q for q in questions if q["expected_pages"]]
    controls = [q for q in questions if not q["expected_pages"]]

    hit_count = 0
    reciprocal_ranks = []
    details = []
    correct_distances = []

    for item in answerable:
        # Compared against printed page labels, which is what a person
        # reads off the document when writing the question set.
        expected = {str(x) for x in item["expected_pages"]}
        hits = search(collection, item["question"], embedder, top_k=k)

        rank = None
        for position, hit in enumerate(hits, start=1):
            if hit["label"] in expected:
                rank = position
                correct_distances.append(hit["distance"])
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
                "returned_pages": [h["label"] for h in hits],
            }
        )

    control_details = []
    control_distances = []
    for item in controls:
        hits = search(collection, item["question"], embedder, top_k=k)
        top = hits[0]["distance"] if hits else None
        if top is not None:
            control_distances.append(top)
        control_details.append(
            {
                "question": item["question"],
                "top_distance": top,
                "returned_pages": [h["label"] for h in hits],
            }
        )

    total = len(answerable)
    return {
        "recall": hit_count / total if total else 0.0,
        "mrr": sum(reciprocal_ranks) / total if total else 0.0,
        "answerable": total,
        "controls": len(controls),
        "details": details,
        "control_details": control_details,
        "correct_distances": correct_distances,
        "control_distances": control_distances,
    }


def print_calibration(scores: dict) -> None:
    """Show the two distance bands and where a threshold could sit.

    A clean separation means one number can serve as a refusal
    threshold. Overlapping bands mean it cannot, and that is a finding
    worth recording rather than a number worth forcing.
    """
    correct = scores["correct_distances"]
    control = scores["control_distances"]
    if not correct or not control:
        return

    print("\nthreshold calibration")
    print(f"  correct hits   {min(correct):.3f} .. {max(correct):.3f}")
    print(f"  controls       {min(control):.3f} .. {max(control):.3f}")

    if max(correct) < min(control):
        suggested = (max(correct) + min(control)) / 2
        print(f"  bands separate -> MAX_DISTANCE = {suggested:.3f}")
    else:
        print("  bands OVERLAP -> no single threshold separates them.")
        print("  Record this in Known limitations instead of forcing a value.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="PDF to evaluate against. Defaults to the only file in data/.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS,
        help="Question set to evaluate against, matching the PDF's language.",
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

    questions = load_questions(args.questions)
    print(f"Document: {pdf_path.name}   Questions: {args.questions.name}")

    results = {}
    for model_name in args.models:
        print(f"Evaluating {model_name} ...")
        embedder = Embedder(model_name)
        collection, chunk_count = build_temp_collection(
            model_name, pdf_path, embedder
        )
        results[model_name] = evaluate(collection, embedder, questions, args.k)
        results[model_name]["chunks"] = chunk_count

    first = next(iter(results.values()))
    print(
        f"\nAnswerable: {first['answerable']}   Controls: {first['controls']}"
        f"   Cut-off: k={args.k}"
    )
    print(f"\n{'model':<50} {'recall@' + str(args.k):>10} {'MRR':>8}")
    print("-" * 70)
    for model_name, scores in results.items():
        print(f"{model_name:<50} {scores['recall']:>10.3f} {scores['mrr']:>8.3f}")

    for model_name, scores in results.items():
        if len(results) > 1:
            print(f"\n=== {model_name} ===")
        print_calibration(scores)

    if args.verbose:
        for model_name, scores in results.items():
            print(f"\n--- {model_name} ---")
            for detail in scores["details"]:
                status = f"rank {detail['rank']}" if detail["rank"] else "MISS"
                print(f"  [{status:>7}] {detail['question']}")
                print(f"            returned: {detail['returned_pages']}")
            for detail in scores["control_details"]:
                top = detail["top_distance"]
                shown = f"{top:.3f}" if top is not None else "n/a"
                print(f"  [control] {detail['question']}")
                print(f"            nearest {shown}  pages {detail['returned_pages']}")


if __name__ == "__main__":
    main()
