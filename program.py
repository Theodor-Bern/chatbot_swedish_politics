"""
program.py — the PROGRAM layer: the parties' adopted party programmes (PDF).

A party programme is the party's own, currently valid statement of ideology
and principles, adopted by its congress. It's where a party says what it IS
("ett socialkonservativt parti med nationalistisk grundsyn"), which is what
broad questions like "vilken ideologi har partierna?" need.

Every PDF is laid out differently, so a heading is recognised by any of:
larger font, a font other than the body font, bold, or numbering ("2.1 …",
"KAPITEL 3."). A wrong heading only makes a context line less precise; the
text itself is unaffected.

Run:
    python program.py build      ->  out/program_chunks.jsonl
    python program.py inspect    ->  headings found per programme
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

PDF_DIR = Path("data/pdf")
OUT = Path("out/program_chunks.jsonl")

# file -> (party, document type, year adopted). The year is when the programme
# was adopted, not a sign that it's outdated: parties replace programmes rarely
# (C still uses its 2013 idéprogram). None = not known.
PROGRAMS = {
    "socialdemokraterna_partiprogram.pdf": ("S", "partiprogram", 2025),
    "moderaterna_partiprogam.pdf": ("M", "idéprogram", 2021),
    "sverigedemokraternas_partiprogram.pdf": ("SD", "principprogram", 2023),
    "centerpartiet_partiprogram.pdf": ("C", "idéprogram", 2013),
    "vansterpartiet_partiprogram.pdf": ("V", "partiprogram", 2024),
    "kristdemokraterna_partiprogram.pdf": ("KD", "principprogram", 2025),
    "miljopartiet_partiprogram.pdf": ("MP", "partiprogram", 2025),
    "liberalerna_partiprogram.pdf": ("L", "partiprogram", 2025),
}

NUMBERED_HEADING = re.compile(r"^(KAPITEL\s+\d+\.?|\d{1,2}(\.\d+)*\.?)\s+[A-ZÅÄÖa-zåäö]")
TOC_TITLE = re.compile(r"^\s*(innehåll|innehållsförteckning)\s*$", re.IGNORECASE)
NOISE = re.compile(r"^(informationsklass:.*|\d{1,3})$", re.IGNORECASE)
LETTERS = re.compile(r"[A-Za-zÅÄÖåäöÉé]{2}")
INVISIBLE = dict.fromkeys(map(ord, "​­﻿"))    # zero-width space, soft hyphen, BOM


def is_bold(span):
    return "bold" in span["font"].lower() or bool(span["flags"] & 16)


# KD's heading font (Gill Sans) is extracted with a space after every "m":
# "Social m arknadsekonom i". Only in this font, so the repair is limited to it.
SPLIT_M_FONTS = ("GillSansProforRiksdag",)


def span_text(span):
    text = span["text"]
    if span["font"].startswith(SPLIT_M_FONTS):
        text = re.sub(r"m (?=[a-zåäö])", "m", text)
    return text


def line_text(line):
    return "".join(span_text(s) for s in line["spans"]).translate(INVISIBLE).strip()


def body_style(doc):
    """The (font, size) that carries the most characters = body text."""
    count = Counter()
    for page in doc:
        for b in page.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                for s in line["spans"]:
                    count[(s["font"], round(s["size"], 1))] += len(s["text"].strip())
    return count.most_common(1)[0][0]


def line_style(line, body_font, body_size):
    """'heading' (styled like one), 'body', 'skip', or ('dropcap', letter).
    Style only — whether a heading-styled block really is a heading is
    decided per block, in is_heading_text()."""
    spans = [s for s in line["spans"] if s["text"].translate(INVISIBLE).strip()]
    text = line_text(line)
    if not spans or NOISE.match(text):
        return "skip"                           # page numbers, footers
    size = max(s["size"] for s in spans)
    if size > body_size * 3:                    # decoration: drop caps, big quote marks
        letter = text.split()[-1]
        return ("dropcap", letter) if re.fullmatch(r"[A-ZÅÄÖ]", letter) else "skip"
    if not LETTERS.search(text) or size < body_size * 0.75:
        return "skip"                           # bullet characters, tiny print
    styled = (size >= body_size * 1.15
              or (spans[0]["font"] != body_font and size > body_size)
              or all(is_bold(s) for s in spans)
              or NUMBERED_HEADING.match(text))
    return "heading" if styled else "body"


def is_heading_text(text):
    """Pull quotes are set like headings but read like sentences: long,
    starting in lower case, or ending with a full stop."""
    if not (2 <= len(text) <= 90) or not LETTERS.search(text):
        return False
    if not (text[0].isupper() or text[0].isdigit()):
        return False
    return not text.endswith(".") or bool(NUMBERED_HEADING.match(text))


def join_lines(lines):
    """Lines of one block -> a paragraph, undoing end-of-line hyphenation."""
    out = ""
    for line in lines:
        if out.endswith("-") and line[:1].islower() and not line.startswith(("och ", "eller ")):
            out = out[:-1] + line               # "demo-" + "krati" -> "demokrati"
        else:
            out = f"{out} {line}".strip()
    return out


def read_program(path):
    """PDF -> [{'heading', 'page', 'text'}], one entry per heading."""
    import pymupdf
    doc = pymupdf.open(path)
    body_font, body_size = body_style(doc)

    sections = []
    current = {"heading": "", "page": 1, "paragraphs": []}
    dropcap = ""

    def start_heading(text, pno):
        nonlocal current
        if current["paragraphs"]:
            sections.append(current)
            current = {"heading": text, "page": pno, "paragraphs": []}
        else:                                   # heading right after heading: merge,
            joined = f"{current['heading']} {text}".strip()   # unless it's numbered:
            # "KAPITEL 1. …" or "2.4.1 …" starts its own section, and the text
            # before it is a chapter title or leftover table-of-contents lines.
            replace = len(joined) > 120 or NUMBERED_HEADING.match(text)
            current["heading"] = text if replace else joined
            current["page"] = pno

    for pno, page in enumerate(doc, 1):
        blocks = []
        for b in page.get_text("dict")["blocks"]:
            if b["type"] == 0:
                styled = [(line_style(l, body_font, body_size), line_text(l)) for l in b["lines"]]
                blocks.append([(k, t) for k, t in styled if k != "skip"])
        page_texts = [t for block in blocks for _, t in block]
        n_headings = sum(k == "heading" for block in blocks for k, _ in block)
        body_chars = sum(len(t) for block in blocks for k, t in block if k == "body")
        if any(TOC_TITLE.match(t) for t in page_texts) or (n_headings >= 8 and body_chars < 600):
            continue                            # table of contents

        for block in blocks:
            for kind, text in block:
                if isinstance(kind, tuple):
                    dropcap = kind[1]
            lines = [(k, t) for k, t in block if not isinstance(k, tuple)]
            if lines and all(k == "heading" for k, _ in lines) \
                    and not is_heading_text(" ".join(t for _, t in lines)):
                continue                        # a pull quote: repeats the body text

            # Split the block at heading runs: some PDFs (KD) put a sub-heading
            # in the middle of a text block, not at its start.
            body, run = [], []

            def flush_body():
                nonlocal dropcap
                if body:
                    paragraph = join_lines(body)
                    if dropcap:                 # "oderaterna" + drop cap "M"
                        paragraph, dropcap = dropcap + paragraph, ""
                    current["paragraphs"].append(paragraph)
                    body.clear()

            for k, t in lines + [("end", "")]:
                if k == "heading":
                    run.append(t)
                    continue
                if run:
                    if is_heading_text(" ".join(run)):
                        flush_body()
                        start_heading(" ".join(run), pno)
                    else:
                        body.extend(run)        # styled lines inside a paragraph
                    run = []
                if k == "body":
                    body.append(t)
            flush_body()
    if current["paragraphs"]:
        sections.append(current)
    return [{"heading": s["heading"], "page": s["page"],
             "text": "\n\n".join(s["paragraphs"])} for s in sections]


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        for filename, (party, doc_type, year) in PROGRAMS.items():
            path = PDF_DIR / filename
            if not path.exists():
                print(f"  missing: {path}")
                continue
            sections = read_program(path)
            for i, s in enumerate(sections):
                fh.write(json.dumps({
                    "chunk_id": f"{party}:program:{i}",
                    "layer": "program",
                    "parti": party,
                    "doc_type": doc_type,
                    "year": year,
                    "heading": s["heading"],
                    "page": s["page"],
                    "text": s["text"],
                    "source_file": filename,
                }, ensure_ascii=False) + "\n")
            words = sum(len(s["text"].split()) for s in sections)
            print(f"  {party:3} {doc_type:15} {len(sections):3} sections  {words:6} words")
    print(f"wrote {OUT}")


def inspect():
    for filename, (party, _doc_type, _year) in PROGRAMS.items():
        sections = read_program(PDF_DIR / filename)
        print(f"\n== {party}: {len(sections)} sections")
        for s in sections:
            print(f"  p{s['page']:<3} {len(s['text']):6} chars  {s['heading'][:70]}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build":
        build()
    elif cmd == "inspect":
        inspect()
    else:
        sys.exit(__doc__)
