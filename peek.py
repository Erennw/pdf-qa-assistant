"""Print chunk statistics and a slice of chunks for visual inspection.

A development tool, not part of the pipeline. Chunking decisions are
easiest to judge by reading the output, and every change to CHUNK_SIZE,
CHUNK_OVERLAP or the cleaning rules is worth re-reading.

Not named inspect.py on purpose: that shadows a standard library module
and breaks any package that imports it from the project root.

Usage:
    python peek.py data/doc.pdf            # stats only
    python peek.py data/doc.pdf 70 73      # stats plus chunks 70-72
"""

import sys
from pathlib import Path

from rag.chunker import chunk_pages
from rag.loader import load_pdf


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    path = Path(sys.argv[1])
    pages = load_pdf(path)
    chunks = chunk_pages(pages, source=path.name)

    lengths = [len(c["text"]) for c in chunks]
    raw_chars = sum(len(p["text"]) for p in pages)

    print(f"pages       {len(pages)}")
    print(f"raw chars   {raw_chars}")
    print(f"chunks      {len(chunks)}")
    print(f"length      avg {sum(lengths) // len(lengths)}  "
          f"min {min(lengths)}  max {max(lengths)}")
    # Chunk text totals more than the source because of overlap. A wildly
    # larger figure means chunks are duplicating each other.
    print(f"overlap     {sum(lengths) - raw_chars} chars "
          f"({(sum(lengths) - raw_chars) // len(chunks)} per chunk)")

    if len(sys.argv) >= 4:
        start, end = int(sys.argv[2]), int(sys.argv[3])
        for chunk in chunks[start:end]:
            print(f"\n--- page {chunk['page']} | {len(chunk['text'])} chars ---")
            print(chunk["text"])


if __name__ == "__main__":
    main()
