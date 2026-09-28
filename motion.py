"""Parser for Swedish parliamentary motions (motioner).

A motion has two parts we care about:

  yrkande     a single, formal proposal for a decision: "Riksdagen ställer sig
              bakom ... och tillkännager detta för regeringen." Sits as
              <li class="Frslagstext"> in the list under "Förslag till
              riksdagsbeslut".
  section     a heading + the body text under it (numbered Rubrik1/2/3numrerat).
              Holds the arguments, and often a near-verbatim restatement of a
              yrkande in its last sentence.

The HTML has no explicit link between a yrkande and its section — only that
the text repeats. We find the link through word overlap: which section shares
the most uncommon words with the yrkande.

Run:
    python motion.py inspect <one motion>.html
    python motion.py build data/html/mot-2022-2025.html out/motion_chunks.jsonl

The output is read by rag_index.py (the MOTION layer) — search and answering
happen there and in answer.py, not here. The output's JSON field names
(parti, sektion_text, yrkande_nr, undertecknare, …) are its data format.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup

# Below this threshold we don't trust the word-overlap match — better to save
# the yrkande without a section than to guess wrong. Set loosely from two
# motions (correct matches were at 85-96%, one observed mismatch at 33%) —
# raise/lower once more motions have been checked.
MIN_MATCH = 0.5

# .Frslagstext, .Hemstlatt and .Yrkande only share CSS formatting in the Aspose
# export (see the <style> block) — nothing says they mean the same thing, but
# in practice all three mark a yrkande line.
YRKANDE_CLASSES = {"Frslagstext", "Hemstlatt", "Yrkande"}
HEADING_CLASSES = {"Rubrik1numrerat", "Rubrik2numrerat", "Rubrik3numrerat", "Rubrik4numrerat"}
# Newer exports (with -aw-sdt-tag-wrapped divs) use real <h1>-<h6> instead of
# <p class="RubrikXnumrerat"> — same role, different markup.
HEADING_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6"]

BETECKNING_RE = re.compile(r"^(?P<rm>\d{4}/\d{2}):(?P<nr>\S+)")
# "(V)" — most common. "(båda C)"/"(alla S)" — when several co-signatories
# share a party, the Riksdag spells it out before the party code.
PARTY_RE = re.compile(r"\((?:[a-zåäö]+\s+)?([A-ZÅÄÖ]{1,3})\)\s*$")

WORD_RE = re.compile(r"[a-zåäöA-ZÅÄÖ0-9]+")
# Swedish stopwords for the word-overlap match (content stays Swedish).
STOPWORDS = set("""och att det som en för av på är den till med de i om så har inte
den där vi men kan ska har hade blir blev vara var vid ett den denna detta dessa
under efter mot från utan även samt då när där vilket vilka man sig sin sitt
dess deras våra vår vårt eller än mer mest också bara alla andra bör detta
riksdagen regeringen till känna ställer sig bakom anförs motionen""".split())


def clean_text(element):
    text = element.get_text()
    text = text.replace("\xad", "").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def read_html(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


# --------------------------------------------------------------------------
# document metadata
# --------------------------------------------------------------------------

def document_info(soup):
    """rm, beteckning, party, first signatory, title."""
    info = {"rm": "", "beteckning": "", "party": "", "first_signatory": "",
            "doc_title": ""}

    beteckning_span = soup.find("span", class_="sidhuvud_beteckning")
    if beteckning_span:
        m = BETECKNING_RE.match(beteckning_span.get_text(strip=True))
        if m:
            info["rm"] = m.group("rm")
            info["beteckning"] = m.group("nr")

    author_span = soup.find("span", class_="MotionarLista")
    if author_span:
        author_text = author_span.get_text(strip=True)
        info["first_signatory"] = author_text
        m = PARTY_RE.search(author_text)
        if m:
            info["party"] = m.group(1)

    title = soup.find("h1")
    if title:
        info["doc_title"] = clean_text(title)

    return info


def extract_signatories(soup):
    """Names of all signatories, from the signature table at the bottom."""
    names = []
    for p in soup.find_all("p", class_="Underskrifter"):
        text = clean_text(p)
        if text:
            names.append(text)
    return names


# --------------------------------------------------------------------------
# yrkanden
# --------------------------------------------------------------------------

def extract_yrkanden(soup):
    """One string per yrkande, in order.

    The yrkande sits as <li class="Frslagstext"> in older exports (an <ol>
    list under "Förslag till riksdagsbeslut"), but as a standalone
    <p class="Frslagstext"> in newer, -aw-sdt-tag-based exports — no <ol>/<li>
    at all, just a <div> with -aw-sdt-title "Yrkande N" around an ordinary
    paragraph. Same class, different tag.
    """
    yrkanden = []
    for el in soup.find_all(["li", "p"]):
        if set(el.get("class", []) or []) & YRKANDE_CLASSES:
            text = clean_text(el)
            if text:
                yrkanden.append(text)
    return yrkanden


# --------------------------------------------------------------------------
# heading sections
# --------------------------------------------------------------------------

def extract_sections(soup):
    """[{'heading': str, 'body': [str, ...]}, ...] — one entry per heading,
    with all body text up to the next heading.

    A heading is either a <p class="RubrikXnumrerat"> (older exports) or a
    real <h1>-<h6> (newer, -aw-sdt-tag-based exports). Both mark the same
    thing — where the body text belongs — just with different markup.

    <li> is skipped: those belong to the yrkande list, not to a section's
    body text, even when they happen to sit inside Section1.
    """
    section1 = soup.find("div", class_="Section1")
    if section1 is None:
        return []

    sections = []
    current = None
    for el in section1.find_all(["p"] + HEADING_TAGS, recursive=True):
        if el.find_parent("li"):
            continue
        classes = set(el.get("class", []) or [])
        if classes & YRKANDE_CLASSES:
            continue    # a standalone <p class="Frslagstext">, not body text
        text = clean_text(el)
        if not text:
            continue
        is_heading = el.name in HEADING_TAGS or bool(classes & HEADING_CLASSES)
        if is_heading:
            if current:
                sections.append(current)
            current = {"heading": text, "body": []}
            continue
        if current is not None:
            current["body"].append(text)
    if current:
        sections.append(current)
    return sections


# --------------------------------------------------------------------------
# matching: which yrkande belongs to which section?
# --------------------------------------------------------------------------

# Light Swedish stemming (same list as rag_index.py) — cuts inflection endings
# so "landsbygden"/"landsbygd" count as the same word. Ordered longest first,
# and only cuts if at least 4 characters remain, so short words like
# "det"/"stat" are left alone.
SUFFIXES = ["ernas", "arnas", "ornas", "andet", "arna", "erna", "orna",
            "ande", "ende", "aste", "ades", "ade", "are", "ast", "ens",
            "ets", "er", "ar", "or", "en", "et", "na", "as", "es"]
MIN_STEM = 4


def stem(word):
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= MIN_STEM:
            return word[: -len(suffix)]
    return word


def tokenize(text):
    words = (w.lower() for w in WORD_RE.findall(text))
    return {stem(w) for w in words if w not in STOPWORDS and len(w) > 2}


def best_match(yrkande, sections):
    """The section (index) whose body text shares the largest share of the
    yrkande's uncommon words, and that share. (None, 0.0) if nothing shares
    anything at all.
    """
    y_words = tokenize(yrkande)
    if not y_words:
        return None, 0.0

    best_i, best_score = None, 0.0
    for i, sec in enumerate(sections):
        sec_words = tokenize(" ".join(sec["body"]))
        if not sec_words:
            continue
        score = len(y_words & sec_words) / len(y_words)
        if score > best_score:
            best_i, best_score = i, score
    return best_i, best_score


# --------------------------------------------------------------------------
# chunk — what actually gets saved
# --------------------------------------------------------------------------

@dataclass
class Chunk:
    """One chunk = one yrkande, with its matched section as context.
    The field names are the output's JSON keys (data format — keep them)."""

    text: str = ""                # the yrkande, whole and unfiltered — what gets embedded
    heading: str = ""              # the heading of the matched section
    sektion_text: str = ""         # the section's body text — context, not embedded directly
    match_score: float = 0.0       # the word overlap that decided the match (see MIN_MATCH)
    yrkande_nr: int = 0

    parti: str = ""
    rm: str = ""
    beteckning: str = ""
    doc_title: str = ""
    forste_undertecknare: str = ""
    undertecknare: list = field(default_factory=list)
    layer: str = "motion"          # a layer of its own — neither SAID nor DID

    @property
    def chunk_id(self):
        return f"{self.rm}:{self.beteckning}:yrkande:{self.yrkande_nr}"

    def to_dict(self):
        d = asdict(self)
        d["chunk_id"] = self.chunk_id
        d["undertecknare"] = ";".join(self.undertecknare)
        return d


def parse_motion(path):
    """One HTML file -> list of Chunk, one per yrkande in the document."""
    soup = BeautifulSoup(read_html(path), "html.parser")

    info = document_info(soup)
    signatories = extract_signatories(soup)
    yrkanden = extract_yrkanden(soup)
    sections = extract_sections(soup)

    chunks = []
    for n, yrkande in enumerate(yrkanden, 1):
        i, score = best_match(yrkande, sections)
        confident = i is not None and score >= MIN_MATCH
        chunks.append(Chunk(
            text=yrkande,
            heading=sections[i]["heading"] if confident else "",
            sektion_text=" ".join(sections[i]["body"]) if confident else "",
            match_score=score,
            yrkande_nr=n,
            parti=info["party"],
            rm=info["rm"],
            beteckning=info["beteckning"],
            doc_title=info["doc_title"],
            forste_undertecknare=info["first_signatory"],
            undertecknare=signatories,
        ))
    return chunks


# --------------------------------------------------------------------------
# build — run over a whole folder of motions
# --------------------------------------------------------------------------

def build(html_folder, out_path):
    # rglob, not glob: the bulk export can hold one folder per document
    # (e.g. mot-2022-2025.html/hb01fiu1/....html), not flat at the top level.
    files = sorted(Path(html_folder).rglob("*.html"))
    print(f"found {len(files)} .html files under {html_folder}", flush=True)
    if not files:
        sys.exit(f"no .html files found in {html_folder} (including subfolders)")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    uncertain = 0
    crashed = []
    per_party = Counter()

    with open(out_path, "w", encoding="utf-8") as fh:
        for path in files:
            try:
                chunks = parse_motion(path)
            except Exception as e:
                crashed.append((path.name, repr(e)))
                continue
            for c in chunks:
                fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
                written += 1
                per_party[c.parti] += 1
                if c.match_score < MIN_MATCH:
                    uncertain += 1

    print(f"read {len(files)} files, wrote {written} chunks to {out_path}")
    print("  per party: " + "  ".join(f"{p} {n}" for p, n in per_party.most_common()))
    print(f"  without a confident section match (< {MIN_MATCH:.0%}): {uncertain}/{written}")
    if crashed:
        print(f"  CRASHED ({len(crashed)}): {crashed}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def inspect(path):
    soup = BeautifulSoup(read_html(path), "html.parser")

    info = document_info(soup)
    signatories = extract_signatories(soup)
    yrkanden = extract_yrkanden(soup)
    sections = extract_sections(soup)

    print(f"doc:      {info['beteckning']} ({info['rm']})")
    print(f"party:    {info['party']}")
    print(f"title:    {info['doc_title']}")
    print(f"1st name: {info['first_signatory']}")
    print(f"all:      {', '.join(signatories)}")
    print(f"\n{len(yrkanden)} yrkanden, {len(sections)} sections\n")

    for n, yrkande in enumerate(yrkanden, 1):
        i, score = best_match(yrkande, sections)
        match = f"[{score:.0%}] {sections[i]['heading']}" if i is not None else "NO MATCH"
        print(f"{n}. {yrkande[:90]}...")
        print(f"   -> {match}\n")


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    cmd = sys.argv[1]
    if cmd == "inspect":
        inspect(Path(sys.argv[2]))
    elif cmd == "build" and len(sys.argv) == 4:
        build(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
