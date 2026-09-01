"""Clean raw PDF text and split it into retrievable chunks.

Chunk size is the single most consequential knob in a RAG system. Too
small and a chunk loses the context that makes it answerable; too large
and the embedding blurs several topics into one vector. The defaults in
config.py are a starting point, not a conclusion -- eval/run.py is
what settles the argument.
"""

import hashlib
import re
from collections import Counter

from rag import config

# A word broken across a line ending: "yönet-\nmelik" -> "yönetmelik".
HYPHEN_LINEBREAK = re.compile(r"(\w)-\s*\n\s*(\w)")

# Three or more newlines collapse to a paragraph break.
EXTRA_NEWLINES = re.compile(r"\n{3,}")

# Runs of spaces or tabs collapse to one space.
EXTRA_SPACES = re.compile(r"[ \t]{2,}")


def find_running_lines(pages: list[dict], threshold: float = 0.5) -> set[str]:
    """Detect headers and footers that repeat across most pages.

    A line printed on every page -- the document title, a publication
    date, a running footer -- says nothing about the passage it sits in,
    but it does pull every chunk containing it toward the same region of
    embedding space, and it costs tokens on every query.

    The rule is document-agnostic: any line appearing on more than
    `threshold` of the pages is treated as page furniture. Page numbers
    differ per page and so survive, which is what we want -- they are
    the one piece of furniture that carries information.
    """
    counts = Counter()
    for page in pages:
        unique_lines = {line.strip() for line in page["text"].splitlines()}
        counts.update(line for line in unique_lines if line)

    minimum = max(2, int(len(pages) * threshold))
    return {line for line, count in counts.items() if count >= minimum}


def clean_text(text: str, drop_lines: set[str] | None = None) -> str:
    """Normalise whitespace and repair line-broken words.

    PDF text extraction preserves the visual line breaks of the printed
    page, which are meaningless to an embedding model. Removing them
    stops the tokenizer from seeing "yönet" and "melik" as two words.

    Args:
        drop_lines: Lines to remove outright, normally the running
            headers found by find_running_lines().
    """
    text = text.replace("\r\n", "\n").replace("\xa0", " ")

    if drop_lines:
        text = "\n".join(
            line
            for line in text.split("\n")
            if line.strip() not in drop_lines
        )

    text = HYPHEN_LINEBREAK.sub(r"\1\2", text)
    text = EXTRA_SPACES.sub(" ", text)
    text = EXTRA_NEWLINES.sub("\n\n", text)
    return text.strip()


def make_chunk_id(source: str, page: int, text: str) -> str:
    """Build a deterministic ID for a chunk.

    Hashing the content means re-ingesting an unchanged document produces
    the same IDs, so the vector store can recognise it and skip the work.
    Edit one character of the text and the ID changes, which is exactly
    the behaviour we want: changed content is treated as a new chunk.
    """
    fingerprint = f"{source}|{page}|{text}".encode("utf-8")
    return hashlib.sha256(fingerprint).hexdigest()[:32]


def _split_page(text: str, size: int, overlap: int) -> list[str]:
    """Split one page into overlapping windows, preferring clean breaks.

    The window advances by (size - overlap) characters. Instead of
    cutting mid-sentence, the end of each window is pulled back to the
    nearest paragraph break, then sentence end, then space -- but only if
    that break falls in the last third of the window, so a document with
    no punctuation still makes progress.
    """
    if size <= overlap:
        raise ValueError("CHUNK_SIZE must be larger than CHUNK_OVERLAP")

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + size, text_length)

        # Only look for a nicer break if we are not already at the end.
        if end < text_length:
            window = text[start:end]
            earliest_acceptable = int(size * 0.66)

            for separator in ("\n\n", ". ", "\n", " "):
                break_at = window.rfind(separator)
                if break_at >= earliest_acceptable:
                    end = start + break_at + len(separator)
                    break

        piece = text[start:end].strip()
        if len(piece) >= config.MIN_CHUNK_SIZE:
            chunks.append(piece)

        # Stop once the page is consumed. Without this the loop runs once
        # more from (end - overlap) and emits a final chunk that is purely
        # a copy of the previous chunk's tail -- one redundant chunk per
        # page, and a duplicate competing with its own source at query time.
        if end >= text_length:
            break

        # Guard against a zero-width step, which would loop forever.
        next_start = end - overlap
        if next_start > start:
            # Align the overlap to a word boundary. Without this the next
            # chunk can begin mid-word ("...tioners sharing"), which both
            # blurs the embedding and looks broken when the excerpt is
            # shown to the user as a citation.
            #
            # The backward search is capped so that alignment can only
            # ever add a bounded amount of overlap; on text with no
            # spaces in that window we accept one broken word rather
            # than silently shrinking the step.
            LOOKBACK = 40
            boundary = text.rfind(
                " ", max(start + 1, next_start - LOOKBACK), next_start
            )
            if boundary > start:
                next_start = boundary + 1
            start = next_start
        else:
            start = end

    return chunks


def chunk_pages(pages: list[dict], source: str) -> list[dict]:
    """Turn loader output into chunks ready for embedding.

    Args:
        pages: Output of loader.load_pdf().
        source: File name of the PDF, stored so a document can later be
            listed or deleted as a unit.

    Returns:
        A list of {"id", "text", "source", "page", "label", "position"}
        dicts. Chunks never span two pages, which keeps every citation to
        a single page.
    """
    running_lines = find_running_lines(pages)
    chunks: list[dict] = []

    for page in pages:
        cleaned = clean_text(page["text"], drop_lines=running_lines)
        if not cleaned:
            continue

        pieces = _split_page(cleaned, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

        for position, piece in enumerate(pieces):
            chunks.append(
                {
                    "id": make_chunk_id(source, page["page"], piece),
                    "text": piece,
                    "source": source,
                    "page": page["page"],
                    "label": page["label"],
                    "position": position,
                }
            )

    return chunks


if __name__ == "__main__":
    # Quick manual check: python -m rag.chunker data/sample.pdf
    import sys

    from rag.loader import load_pdf

    target = sys.argv[1] if len(sys.argv) > 1 else None
    if not target:
        print("Usage: python -m rag.chunker <path-to-pdf>")
        raise SystemExit(1)

    loaded = load_pdf(target)
    produced = chunk_pages(loaded, source=target.split("/")[-1])
    lengths = [len(c["text"]) for c in produced]

    print(f"Chunks: {len(produced)}")
    print(f"Shortest: {min(lengths)}  Longest: {max(lengths)}")
    print(f"Average: {sum(lengths) // len(lengths)}")
    print("\n--- First chunk ---")
    print(produced[0]["text"])
