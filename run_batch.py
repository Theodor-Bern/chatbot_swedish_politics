"""Run the parser over a folder of betänkanden and report how it did.

Usage:
    python run_batch.py data/html          # whole corpus
    python run_batch.py data/html 20       # first 20 files only
"""

import sys
from collections import Counter
from pathlib import Path

from betankande import (
    RESERVATION, DISSENT, SECTION_BY_CLASS,
    parse_document, parse_heading, expected_counts, actual_counts,
)


def main(folder, limit=None):
    files = [p for p in sorted(Path(folder).rglob("*.html")) if p.is_file()]
    if limit:
        files = files[:limit]
    if not files:
        print(f"No .html files found under {folder}")
        return

    total_chunks = 0
    crashed, empty, unmatched, mismatched = [], [], [], []
    by_party = Counter()

    for path in files:
        try:
            info, blocks, chunks = parse_document(path)
        except Exception as e:
            crashed.append((path.name, repr(e)))
            continue

        total_chunks += len(chunks)
        if not chunks:
            empty.append(path.name)

        # headings the regex could not read
        for css_class, heading, _ in blocks:
            if css_class in SECTION_BY_CLASS and parse_heading(heading) is None:
                unmatched.append((path.name, heading[:70]))

        # checksum: does the summary agree with what we extracted?
        expected = expected_counts(blocks)
        actual = actual_counts(chunks)
        if expected != actual:
            mismatched.append((path.name, expected, actual))

        for c in chunks:
            for party in c.parties:
                by_party[party] += 1

    ok = len(files) - len(crashed) - len(mismatched)
    print(f"{len(files)} files, {total_chunks} chunks")
    print(f"  matching the stated count: {ok} "
          f"({ok / len(files):.1%})")
    print(f"  crashed:            {len(crashed)}")
    print(f"  no chunks:          {len(empty)}")
    print(f"  unmatched headings: {len(unmatched)}")
    print(f"  count mismatches:   {len(mismatched)}")

    if by_party:
        print("\nChunks per party:")
        for party, n in by_party.most_common():
            print(f"    {party:4s} {n:6d}")

    for name, e, a in mismatched[:10]:
        print(f"\n  MISMATCH {name}: summary says {e}, parser found {a}")
    for name, heading in unmatched[:10]:
        print(f"  UNMATCHED {name}: {heading}")
    for name, err in crashed[:5]:
        print(f"  CRASH {name}: {err}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)