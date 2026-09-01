"""Build the grounded prompt and call the LLM.

The retrieval half of RAG decides what the model *can* see. This module
decides what it is *allowed to do* with what it sees -- and that is where
hallucination is actually prevented, in the system prompt, not in the
vector search.
"""

import anthropic

from rag import config
from rag.embedder import Embedder
from rag.store import search

SYSTEM_PROMPT = """You answer questions about a document the user has uploaded.

GROUNDING RULE
Use only the excerpts provided in the CONTEXT block. Your own knowledge
about the subject is not evidence. If the context does not contain the
answer, say so plainly instead of filling the gap.

CITATION RULE
End every factual sentence with its page reference in the form (p. 12).
When a sentence draws on two excerpts, cite both: (p. 12, p. 15).
A sentence without a page reference is not allowed.

REFUSAL RULE
When the context is insufficient, reply with one sentence stating that
the document does not cover this, and suggest what the user might ask
instead. Do not apologise and do not speculate.

STYLE RULE
Answer in the same language as the question. Be direct: three or four
sentences unless the question genuinely requires more. Quote the document
only when exact wording matters, and keep such quotes under fifteen words.
"""

REFUSAL_MESSAGE = (
    "I could not find anything relevant to that in the document. "
    "Try rephrasing the question with wording closer to the text."
)


def build_context(hits: list[dict]) -> str:
    """Format retrieved chunks into a labelled CONTEXT block.

    The page label is repeated on every excerpt so the model can cite it
    without having to count or infer which excerpt it is using. The label
    is the number printed on the page, not the position in the file, so a
    reader who follows the citation lands where they expect.
    """
    blocks = []
    for index, hit in enumerate(hits, start=1):
        blocks.append(
            f"[Excerpt {index} | page {hit['label']}]\n{hit['text']}"
        )
    return "\n\n".join(blocks)


def answer_question(
    collection,
    embedder: Embedder,
    question: str,
    source: str | None = None,
    top_k: int = config.TOP_K,
    max_distance: float = config.MAX_DISTANCE,
) -> dict:
    """Retrieve, ground, and answer.

    Returns:
        {
            "answer": str,
            "hits": list[dict],   # what was retrieved, for the UI
            "grounded": bool,     # False when the LLM was never called
        }

    When nothing clears the distance threshold the LLM is skipped
    entirely. This is both a quality decision (no context, no grounded
    answer) and a cost decision (no tokens spent on a hopeless query).
    """
    hits = search(collection, question, embedder, top_k=top_k, source=source)
    relevant = [h for h in hits if h["distance"] <= max_distance]

    if not relevant:
        return {"answer": REFUSAL_MESSAGE, "hits": hits, "grounded": False}

    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and "
            "fill in your key."
        )

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    user_message = (
        f"CONTEXT\n{build_context(relevant)}\n\n"
        f"QUESTION\n{question}"
    )

    response = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=config.LLM_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    text = "".join(
        block.text for block in response.content if block.type == "text"
    )

    return {"answer": text.strip(), "hits": relevant, "grounded": True}
