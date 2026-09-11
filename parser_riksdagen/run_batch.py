"""Run the parser over a folder of betänkanden and report how it did.

Usage:
    python run_batch.py data/html          # whole corpus
    python run_batch.py data/html 20       # first 20 files only

The summary sentence of each document ("I betänkandet finns åtta
reservationer") is the checksum: no hand-written ground truth needed. A
document is only scored when that sentence makes a claim, so the report has
three buckets, not two — pretending "unverifiable" means "correct" would
flatter the parser, and pretending it means "wrong" hides real accuracy.
"""

import sys
from collections import Counter
from pathlib import Path

from parser_riksdagen.betankande import (
    RESERVATION, DISSENT, SECTION_BY_CLASS,
    read_html, document_kind, parse_document, parse_heading,
    expected_counts, actual_counts, verify,
)


def main(folder, limit=None):
    files = [p for p in sorted(Path(folder).rglob("*.html")) if p.is_file()]
    if limit:
        files = files[:limit]
    if not files:
        print(f"No .html files found under {folder}")
        return

    kinds = Counter()
    verdicts = Counter()
    by_party = Counter()
    by_section = Counter()
    lengths = []

    total_chunks = 0
    crashed, empty, unmatched, failures, unverifiable = [], [], [], [], []

    for path in files:
        raw = read_html(path)
        kind = document_kind(raw)
        kinds[kind] += 1
        if kind != "word":
            continue                    # no structure to parse; not a failure

        try:
            info, paragraphs, blocks, chunks = parse_document(path, raw)
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

        expected = expected_counts(paragraphs)
        actual = actual_counts(chunks)
        verdict = verify(expected, actual)
        verdicts[verdict] += 1
        if verdict == "fail":
            failures.append((path.name, expected, actual))
        elif verdict == "unverifiable":
            unverifiable.append(path.name)

        for c in chunks:
            by_section[c.section] += 1
            lengths.append(c.n_chars)
            for party in c.parties:
                by_party[party] += 1

    # ---- report ----------------------------------------------------------
    scored = verdicts["pass"] + verdicts["fail"]
    parsed = kinds["word"]

    print(f"{len(files)} files")
    for kind, n in kinds.most_common():
        print(f"    {kind:12s} {n:5d}")

    print(f"\n{parsed} parsed -> {total_chunks} chunks")
    print(f"  crashed:                    {len(crashed)}")
    print(f"  unmatched headings:         {len(unmatched)}")
    print(f"  no reservations at all:     {len(empty)}")

    print(f"\nChecksum against the summary sentence:")
    print(f"  pass:                       {verdicts['pass']}")
    print(f"  fail:                       {verdicts['fail']}")
    print(f"  unverifiable (no claim):    {verdicts['unverifiable']}")
    if scored:
        print(f"  accuracy on scored docs:    {verdicts['pass'] / scored:.1%}")
    if parsed:
        print(f"  coverage:                   {scored}/{parsed} = "
              f"{scored / parsed:.1%}")

    if by_section:
        print("\nChunks per section:")
        for section, n in by_section.most_common():
            print(f"    {section:12s} {n:6d}")

    if by_party:
        print("\nChunks per party:")
        for party, n in by_party.most_common():
            print(f"    {party:4s} {n:6d}")

    if lengths:
        lengths.sort()
        def pct(p):
            return lengths[min(int(len(lengths) * p), len(lengths) - 1)]
        print(f"\nChunk length in characters "
              f"(median {pct(0.5)}, p90 {pct(0.9)}, max {lengths[-1]})")
        over = sum(1 for n in lengths if n > 3000)
        print(f"  longer than 3000 chars ({over}) will need splitting "
              f"before embedding")

    for name, e, a in failures[:15]:
        print(f"\n  FAIL {name}: summary claims {e}, parser found {a}")
    for name, heading in unmatched[:10]:
        print(f"  UNMATCHED {name}: {heading}")
    for name, err in crashed[:5]:
        print(f"  CRASH {name}: {err}")
    if unverifiable:
        print(f"\n  UNVERIFIABLE: {', '.join(unverifiable[:10])}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)