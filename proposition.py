"""Parser for the government's propositions (propositioner).

A proposition is hundreds of pages: law text, background, consultation
responses, Lagrådet. For a chatbot about party positions only two parts matter:

  summary   "Propositionens huvudsakliga innehåll": a plain-language summary
            of the whole reform, in every proposition, before the table of
            contents. One record per proposition.
  sections  The budget (:1) and spring (:100) propositions have a summary
            that only says "here is the budget". Their policy sits in the
            Finansplan instead: section 1.1, the "Reformer…" section and
            "Politiska utgångspunkter". One record per topic heading.

The HTML is a PDF export: one <p> per line or paragraph, page headers and
page numbers mixed in, no semantic markup. Everything is done on the text.

Skrivelser (reports to the Riksdag) sit in the same folder and are skipped:
they report, they don't propose.

Run:
    python proposition.py build data/html/prop-2022-2025.html out/proposition_chunks.jsonl

The output is read by rag_index.py (the PROPOSITION layer).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

# The Tidö government (M, KD, L) took office on 18 October 2022.
GOVERNMENT = "M;KD;L"
# ponytail: the two propositions signed by the previous S government
# (2022/23:3 and :6, 13 Oct 2022) are skipped; give them parti "S" if they matter.
PREVIOUS_PM = "Magdalena Andersson"

ID_RE = re.compile(r"Regeringens proposition (\d{4}/\d{2}):(\d+)")
DATE_RE = re.compile(r"Stockholm den (\d{1,2} \w+ \d{4})")
SUMMARY_RE = re.compile(r"\nPropositionens huvudsakliga innehåll\n(.*?)\nInnehållsförteckning\n", re.S)
# A section heading in the body: "1.4\nReformer\n". In the table of contents
# the title is followed by dot leaders, so it doesn't match.
HEADING_RE = re.compile(r"\n(\d\.\d)\n([A-ZÅÄÖ][^\n.]{3,90})\n")
KEEP_TITLE_RE = re.compile(r"^(Reformer|Politiska utgångspunkter)")
# Lines that are page furniture or table cells, not prose.
NOISE_RE = re.compile(r"Prop\.|Prop\. \d{4}/\d{2}:\d+|\d{4}/\d{2}:\d+|[\d\s,.−\-–+%()]+")
# PDF line-break hyphenation, "hus- hållens" -> "hushållens". Not before
# "och/eller/…": "arbets- och familjeliv" is real Swedish.
HYPHEN_RE = re.compile(r"(\w)- (?!och\b|eller\b|samt\b|till\b|som\b)([a-zåäö])")
# Head of the HTML that always holds title, date and summary. Parsing all
# 800 MB takes minutes; the summary sits within the first pages.
HEAD_BYTES = 400_000
MIN_CHARS = 150


def read_text(path, limit=None):
    with open(path, encoding="utf-8", errors="replace") as fh:
        html = fh.read(limit) if limit else fh.read()
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    return title, "\n" + soup.get_text("\n", strip=True) + "\n"


def paragraphs(text):
    """PDF lines -> paragraphs. A line joins the previous one when that one
    didn't end a sentence and either is long (a wrapped line) or this one
    starts in lowercase. A short line without full stop is a heading."""
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or NOISE_RE.fullmatch(line):
            continue
        prev = out[-1] if out else ""
        if prev and not prev.endswith((".", "!", "?", ":")) and (
                len(prev) >= 90 or line[0].islower() or prev.endswith((",", "–", "-"))):
            out[-1] = f"{prev} {line}"
        else:
            out.append(line)
    return [HYPHEN_RE.sub(r"\1\2", p) for p in out]


def is_heading(p):
    return len(p) < 100 and not p.endswith((".", "!", "?", ":", ","))


def mostly_numbers(p):
    return sum(c.isdigit() for c in p) > 0.2 * len(p)


def policy_sections(text, budget):
    """(section title, body text) for the sections that carry policy."""
    heads = list(HEADING_RE.finditer(text))
    taken = set()       # appendices restart at "1.1": only the first counts
    for h, nxt in zip(heads, heads[1:] + [None]):
        number, title = h.group(1), h.group(2).strip()
        if number not in taken and (KEEP_TITLE_RE.match(title) or (budget and number == "1.1")):
            taken.add(number)
            yield title, text[h.end():nxt.start() if nxt else len(text)]


def topic_records(base, section, body):
    """One record per topic heading inside a section; rag_index splits long ones."""
    heading, buf, out = section, [], []
    for p in paragraphs(body) + ["(slut)"]:
        if is_heading(p):
            body = "\n\n".join(buf)
            if len(body) >= MIN_CHARS:      # shorter = diagram captions, table notes
                out.append({**base, "heading": heading, "text": body})
            heading, buf = (p if p != "(slut)" else heading), []
        elif not mostly_numbers(p):
            buf.append(p)
    return out


def parse(path):
    title, text = read_text(path, HEAD_BYTES)
    found = ID_RE.search(text[:5000])
    if not found:       # the biggest files have more than HEAD_BYTES of <style> first
        title, text = read_text(path)
        found = ID_RE.search(text[:5000])
    if not found or title.startswith("TEST") or PREVIOUS_PM in text[:5000]:
        return None, []
    rm, nr = found.groups()
    budget, spring = nr == "1", nr == "100"
    summary = SUMMARY_RE.search(text)
    if budget or spring or not summary:
        title, text = read_text(path)
        summary = SUMMARY_RE.search(text)
    date = DATE_RE.search(text)
    base = {"rm": rm, "beteckning": nr, "doc_title": title, "parti": GOVERNMENT,
            "datum": date.group(1) if date else "", "file": path.name}

    if budget or spring:
        records = [r for section, body in policy_sections(text, budget)
                   for r in topic_records(base, section, body)]
    elif summary:
        records = [{**base, "heading": "Propositionens huvudsakliga innehåll",
                    "text": "\n\n".join(paragraphs(summary.group(1)))}]
    else:
        records = []
    for i, r in enumerate(records):
        r["chunk_id"] = f"prop:{rm}:{nr}:{i}"
    return (rm, nr), records


def build(html_folder, out_path):
    seen, n_docs, out = set(), 0, []
    files = sorted(Path(html_folder).glob("*.html"))
    for i, path in enumerate(files, 1):
        key, records = parse(path)
        if key and key not in seen and records:
            seen.add(key)
            n_docs += 1
            out.extend(records)
        elif key and key not in seen:
            print(f"  no text extracted: {path.name} {key[0]}:{key[1]}")
        if i % 100 == 0:
            print(f"  {i}/{len(files)} files")
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{n_docs} propositions -> {len(out)} records -> {out_path}")


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "build":
        build(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
