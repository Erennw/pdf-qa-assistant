"""Turn a PDF file into plain text, one entry per page.

This module knows nothing about chunking or embeddings. Its only job is
PDF in, text out -- which is what makes it easy to swap pypdf for an OCR
backend later without touching the rest of the pipeline.
"""

import re
from pathlib import Path

from pypdf import PdfReader

# A line that is nothing but a page number, printed on its own.
ARABIC_ONLY = re.compile(r"^\d{1,4}$")
ROMAN_ONLY = re.compile(r"^[ivxlcdm]{1,7}$", re.IGNORECASE)


def _printed_label(text: str) -> str | None:
    """Find a standalone page number among a page's first and last lines.

    Page numbers sit in the header or footer, so only the outermost few
    lines are considered. A line deep in the body that happens to be a
    bare number is not a page number.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:3] + lines[-3:]:
        if ARABIC_ONLY.match(line) or ROMAN_ONLY.match(line):
            return line
    return None


def _detect_printed_labels(reader: PdfReader) -> list[str] | None:
    """Read printed page numbers off the pages themselves.

    Most PDFs do not declare /PageLabels, yet almost all of them print
    the number on the page. Extracting it is a heuristic, so the result
    is only accepted when it behaves like real pagination: found on most
    pages, and strictly increasing where the numbers are arabic.

    Returning None means "could not establish this reliably", which the
    caller treats as a reason to fall back rather than to guess.
    """
    candidates = [_printed_label(page.extract_text() or "") for page in reader.pages]

    found = [c for c in candidates if c is not None]
    if len(found) < len(candidates) * 0.6:
        return None

    numbers = [int(c) for c in found if ARABIC_ONLY.match(c)]
    if len(numbers) < 3 or any(b <= a for a, b in zip(numbers, numbers[1:])):
        return None

    # Pages where nothing was found keep their file position, which is
    # visibly wrong rather than quietly wrong.
    return [c if c else str(i) for i, c in enumerate(candidates, start=1)]


class EmptyPdfError(Exception):
    """Raised when a PDF yields no extractable text at all.

    The usual cause is a scanned document: the pages are images, so there
    are no text layers for pypdf to read. Such a file needs OCR before it
    can enter this pipeline.
    """


def load_pdf(path: str | Path) -> list[dict]:
    """Read a PDF and return its pages as dictionaries.

    Args:
        path: Path to the PDF file.

    Returns:
        A list of {"page": int, "label": str, "text": str} entries with
        blank pages skipped.

        "page" is the position in the file, 1-indexed. "label" is the
        number printed on the page itself, read from the PDF's own page
        labels. The two differ whenever a document has front matter: in
        a report numbered i-iv before page 1, file page 10 is printed
        page 5. Citing the file position would send a reader to the
        wrong place, so "label" is what the user is shown.

    Raises:
        FileNotFoundError: The path does not exist.
        EmptyPdfError: No page produced any text.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No such PDF: {path}")

    reader = PdfReader(str(path))

    # Prefer the PDF's declared labels; they are authoritative when
    # present. pypdf synthesises 1..N when nothing is declared, so check
    # the catalog rather than trusting a non-empty return -- otherwise
    # the fallback below would never run.
    labels = None
    try:
        if "/PageLabels" in reader.root_object:
            labels = reader.page_labels
    except Exception:
        labels = None

    # Otherwise read the numbers printed on the pages themselves. Only
    # if that is inconclusive does the file position stand in.
    if not labels:
        labels = _detect_printed_labels(reader)

    # Some PDFs are encrypted with an empty owner password. Trying an
    # empty string unlocks those without bothering the user.
    if reader.is_encrypted:
        reader.decrypt("")

    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            if labels and page_number <= len(labels):
                label = str(labels[page_number - 1])
            else:
                label = str(page_number)
            pages.append({"page": page_number, "label": label, "text": text})

    if not pages:
        raise EmptyPdfError(
            f"{path.name} contains no extractable text. "
            "It is probably a scanned document and needs OCR first."
        )

    return pages


if __name__ == "__main__":
    # Quick manual check: python -m rag.loader data/sample.pdf
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else None
    if not target:
        print("Usage: python -m rag.loader <path-to-pdf>")
        raise SystemExit(1)

    loaded = load_pdf(target)
    total_chars = sum(len(p["text"]) for p in loaded)
    print(f"Pages with text: {len(loaded)}")
    print(f"Total characters: {total_chars}")
    print(f"Average per page: {total_chars // len(loaded)}")
    print("\n--- First 400 characters of page 1 ---")
    print(loaded[0]["text"][:400])