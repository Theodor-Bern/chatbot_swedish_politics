"""Parse the corpus once and write it to disk.

Usage:
    python build_corpus.py data/html out/chunks.jsonl

Everything downstream — splitting, embedding, retrieval, evaluation — reads
the JSONL, so re-parsing becomes a deliberate step instead of a two-minute
tax on every experiment. A report is written alongside it: the corpus file
is useless without knowing which parser produced it and how it scored.

JSONL rather than parquet on purpose: no extra dependency, greppable, and
the diff between two builds is readable.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

from parser_riksdagen.betankande import (
    RESERVATION, DISSENT, SECTION_BY_CLASS,
    read_html, document_kind, parse_document, parse_heading,
    expected_counts, actual_counts, verify, check_decision_refs,
)

# "AU6" -> "AU", "FöU1" -> "FöU". The committee is the beteckning's prefix,
# so the CSV manifest is not needed for it either.
UTSKOTT_RE = re.compile(r"^([A-ZÅÄÖ][A-Za-zåäö]*?)(?=\d)")


def utskott_of(beteckning):
    m = UTSKOTT_RE.match(beteckning or "")
    return m.group(1) if m else ""


def build(html_folder, out_path):
    files = [p for p in sorted(Path(html_folder).rglob("*.html")) if p.is_file()]
    if not files:
        print(f"No .html files found under {html_folder}")
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kinds = Counter()
    verdicts = Counter()
    by_section = Counter()
    lengths = []
    written = 0
    agree_total = disagree_total = 0
    crashed, failures, unverifiable = [], [], []

    with open(out_path, "w", encoding="utf-8") as fh:
        for path in files:
            raw = read_html(path)
            kind = document_kind(raw)
            kinds[kind] += 1
            if kind != "word":
                continue

            try:
                info, paragraphs, blocks, chunks = parse_document(path, raw)
            except Exception as e:
                crashed.append((path.name, repr(e)))
                continue

            verdict = verify(expected_counts(paragraphs), actual_counts(chunks))
            verdicts[verdict] += 1
            if verdict == "fail":
                failures.append(path.name)
            elif verdict == "unverifiable":
                unverifiable.append(path.name)

            agree, disagree = check_decision_refs(chunks)
            agree_total += agree
            disagree_total += disagree

            utskott = utskott_of(info["beteckning"])
            for n, c in enumerate(chunks):
                if not c.text.strip():
                    continue        # a heading with no body under it
                c.utskott = utskott
                row = c.to_dict()
                # Stable across rebuilds as long as document order is stable,
                # which it is: chunks come out in document order.
                row["chunk_id"] = f"{c.dok_id}:{c.section}:{n}"
                # The checksum verdict travels with every chunk, so a later
                # experiment can exclude documents the parser is unsure about.
                row["doc_verdict"] = verdict
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                by_section[c.section] += 1
                lengths.append(c.n_chars)

    # ---- report ---------------------------------------------------------
    report = []
    add = report.append
    add(f"source:  {html_folder}")
    add(f"output:  {out_path}  ({written} chunks)")
    add("")
    add("files:")
    for kind, n in kinds.most_common():
        add(f"    {kind:12s} {n:5d}")
    add("")
    add("chunks per section:")
    for section, n in by_section.most_common():
        add(f"    {section:12s} {n:6d}")
    add("")

    scored = verdicts["pass"] + verdicts["fail"]
    add("checksum against the summary sentence:")
    add(f"    pass          {verdicts['pass']}")
    add(f"    fail          {verdicts['fail']}")
    add(f"    unverifiable  {verdicts['unverifiable']}")
    if scored:
        add(f"    accuracy      {verdicts['pass'] / scored:.1%}")
    if kinds["word"]:
        add(f"    coverage      {scored}/{kinds['word']} = "
            f"{scored / kinds['word']:.1%}")
    add("")

    refs = agree_total + disagree_total
    if refs:
        add("decision-list reservation refs vs reservation headings:")
        add(f"    agree {agree_total}/{refs} = {agree_total / refs:.1%}")
        add("")

    if lengths:
        lengths.sort()

        def pct(p):
            return lengths[min(int(len(lengths) * p), len(lengths) - 1)]

        add(f"chunk length: median {pct(0.5)}, p90 {pct(0.9)}, "
            f"max {lengths[-1]}")
        add(f"    over 3000 chars (need splitting): "
            f"{sum(1 for n in lengths if n > 3000)}")
        add("")

    if crashed:
        add(f"CRASHED: {crashed}")
    if failures:
        add(f"checksum failures: {', '.join(failures)}")
    if unverifiable:
        add(f"unverifiable: {', '.join(unverifiable)}")

    text = "\n".join(report)
    print(text)
    (out_path.parent / "corpus_report.txt").write_text(text + "\n",
                                                       encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    build(sys.argv[1], sys.argv[2])