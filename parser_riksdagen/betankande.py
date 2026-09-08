"""Parser for Swedish parliamentary committee reports (betänkanden).

Turns riksdagen's HTML export into Chunk records ready for embedding.

GLOSSARY — Swedish domain terms kept untranslated on purpose, because they
name institutional concepts and must match the source data and riksdagen's
own field names:

  betänkande       committee report; the document type we parse
  utlåtande        committee opinion on an EU proposal; same structure,
                   different word in the summary sentence
  reservation      formal dissent by one or more parties; goes to a vote
  motivreservation dissent on the reasoning only, not the decision
  yttrande         "särskilt yttrande", a weaker dissent that is NOT voted on
  punkt            numbered decision item; the join key to voting records
  ställningstagande  the committee majority's reasoning on a punkt
  beteckning       report designation, e.g. "FiU1"
  rm               "riksmöte", the parliamentary session, e.g. "2022/23"
  dok_id           riksdagen's document identifier, e.g. "HA01FIU1"
"""

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

from bs4 import BeautifulSoup


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

SUMMARY = "summary"
DECISION = "decision"
DELIBERATION = "deliberation"
RESERVATION = "reservation"
DISSENT = "dissent"

# Headings that start a new block. Grupprubrik is deliberately absent: it
# marks lettered sub-items a), b) that belong to a single decision.
BOUNDARIES = {
    "Sammanfattning",
    "Avsnittsrubrik",
    "Reservationsrubrik",
    "Srskiltyttranderubrik",
}

# Everything from here on is appendix material.
STOP_AT = {"Bilaga", "Bilagerubrik"}

SECTION_BY_CLASS = {
    "Reservationsrubrik": RESERVATION,
    "Srskiltyttranderubrik": DISSENT,
}

# Appendix data tables start around 16 cells. Word also uses tiny tables to
# hang reservation numbers in the margin, and those must be kept.
DATA_TABLE_MIN_CELLS = 8

PARTY = r"[A-ZÅÄÖ]{1,3}"

# "Riktlinjerna ..., punkt 1 (S)"
# "Statens budget ... (MP)"
# "Vårändringsbudget ..., punkt 1 (V, MP) – utrikes underrättelsetjänst"
HEADING_RE = re.compile(
    rf"^(?P<heading>.+?)"
    rf"(?:,\s*punkt(?:erna)?\s+(?P<punkter>\d+(?:\s*(?:,|och)\s*\d+)*))?"
    rf"\s*\((?P<parties>{PARTY}(?:\s*,\s*{PARTY})*)\)"
    rf"(?P<tail>\s*[–—-]\s*.+)?$"
)

# Reservations start "av Name (S), ...", dissents start "Name (S), ... anför:"
SIGNATORY_RE = re.compile(rf"^(av\s+)?[A-ZÅÄÖ][\w.\- ]+\({PARTY}\)")

# A stray "2." at the end of a body belongs to the next reservation.
BARE_NUMBER_RE = re.compile(r"^\d+\.$")

# ---- Document triage ------------------------------------------------------
# 48 of the 1482 files in the dump carry no parseable structure. Counting them
# as parser failures hides the real accuracy.

# No real betänkande is this small. 34 files are shells of one kind or
# another: "Dokumentet är inte publicerat", a bare <h1> title, "Se bifogad
# pdf.", a link to a sign-language debate. The distinction does not matter.
MIN_DOCUMENT_BYTES = 2000

# PDF-to-HTML exports declare per-page CSS rules, "#page_1 {position:absolute…".
# Note this is a CSS selector in a <STYLE> block, not an id= attribute.
PDFSCAN_RE = re.compile(r"#page_\d")

# ---- Document identity ----------------------------------------------------
DOC_TITLE_CLASS = "DokumentRubrik"
DOKBET_CLASS = "Dokumentbeteckning"

# "2022/23:AU6", "2023/24:FöU1" — rm and beteckning, the join key to the
# voting records. Read from the document rather than from the filename: the
# FöU files were unpacked with the wrong filename codec and their stems are
# mojibake ("HA01F”U1" for "HA01FÖU1").
DOKBET_RE = re.compile(r"^(?P<rm>\d{4}/\d{2}):(?P<beteckning>\S+)")

# ---- Locating the summary -------------------------------------------------
# The summary heading is class "Sammanfattning" in most documents, plain "R2"
# in 92 of them, and entirely absent in 21. Match on text, not on class.
SUMMARY_HEADINGS = {"sammanfattning", "inledning och sammanfattning"}

# Whichever of these comes first ends the summary.
SUMMARY_ENDS = {
    "behandlade förslag",
    "innehållsförteckning",
    "utskottets förslag till riksdagsbeslut",
    "redogörelse för ärendet",
}

# Shorter than this is a Word export glitch, not a summary. hb01sku12 has its
# summary body misfiled into the table of contents at export time.
MIN_SUMMARY_CHARS = 40

# ---- Reading the counts the summary states --------------------------------
# A sentence ends at a period NOT followed by a letter, so that Swedish
# abbreviations ("bl.a.", "m.m.", "t.ex.") do not truncate the clause.
SENTENCE_TAIL = r"(?:[^.]|\.(?=[\wåäö]))*"

# "I betänkandet finns ...", "Betänkandet innehåller ...",
# "I utlåtandet finns ..."
CLAUSE_RE = re.compile(
    rf"\b(?:finns|innehåller)\b(?P<body>{SENTENCE_TAIL})",
    re.IGNORECASE,
)

# Both halves of "två reservationer (V, C) och två särskilda yttranden (S, MP)".
# Longer alternatives first, so "motivreservation" is not matched as
# "reservation" with a truncated count.
KIND_RE = re.compile(
    r"(?P<count>\d+|[a-zåäö]+)\s+"
    r"(?P<kind>motivreservationer|motivreservation|reservationer|reservation"
    r"|särskilda\s+yttranden|särskilt\s+yttrande)",
    re.IGNORECASE,
)

# Used to tell "the summary never mentions this kind" — which means there are
# none — from "it mentions it without counting", which cannot be verified.
KIND_WORDS = {
    RESERVATION: re.compile(r"reservation", re.IGNORECASE),
    DISSENT: re.compile(r"särskilt\s+yttrande|särskilda\s+yttranden",
                        re.IGNORECASE),
}

NUMBER_WORDS = {
    "en": 1, "ett": 1, "två": 2, "tre": 3, "fyra": 4, "fem": 5,
    "sex": 6, "sju": 7, "åtta": 8, "nio": 9, "tio": 10, "elva": 11,
    "tolv": 12, "tretton": 13, "fjorton": 14, "femton": 15,
    "sexton": 16, "sjutton": 17, "arton": 18, "nitton": 19, "tjugo": 20,
}

# ---- The majority side: decisions and deliberations -----------------------
# Top-level sections, identified by heading text. Class is unreliable here:
# "Utskottets ställningstagande" is R3 in most documents and unstyled in some.
TOP_LEVEL_HEADINGS = {
    "utskottets förslag till riksdagsbeslut",
    "redogörelse för ärendet",
    "utskottets överväganden",
    "reservationer",
    "särskilda yttranden",
    "särskilt yttrande",
}

DECISION_HEADING = "utskottets förslag till riksdagsbeslut"
DELIBERATION_HEADING = "utskottets överväganden"
STANDPOINT_HEADING = "utskottets ställningstagande"
KORTHET_HEADING = "utskottets förslag i korthet"

# These three are perfectly consistent across the corpus.
KORTHET_CLASS = "Normalrutafet"       # the "förslag i korthet" box heading
KORTHET_BODY_CLASS = "Normalruta"     # its one-line statement
COMPARE_CLASS = "Normalrutaindrag"    # "Jämför reservation 1 (S), 2 (C)…"
RES_REF_CLASS = "Reservationshnvisning"   # "Reservation 1 (S)" in the
                                          # decision list, hung in the margin

# The punkt number "1." in the decision list is NormalLeft in some documents
# and unstyled in others, so it is found by text.
PUNKT_NUMBER_CLASSES = {"NormalLeft", ""}

# Classes that start a new block. A ställningstagande or a decision item ends
# here — without this, a document missing its expected sub-structure lets one
# chunk swallow the rest of the section (ha01ku32 produced a single 228,000
# character decision chunk). R4 is deliberately absent: it marks a sub-heading
# *inside* a section and is part of its content.
BLOCK_HEADING_CLASSES = {
    "R2", "R3", "Grupprubrik", "Avsnittsrubrik",
    "Normalrutafet", "Reservationsrubrik", "Srskiltyttranderubrik",
}

NUMBER_RE = re.compile(r"\d+")

# What context_line() calls each majority-side section.
SECTION_LABELS = {
    DECISION: "Utskottets förslag till riksdagsbeslut",
    SUMMARY: "Utskottets förslag i korthet",
    DELIBERATION: "Utskottets ställningstagande",
}


# --------------------------------------------------------------------------
# The output record
# --------------------------------------------------------------------------

@dataclass
class Chunk:
    """One unit in the vector index."""

    # Content
    text: str = ""                    # what gets embedded
    heading: str = ""                 # what the chunk is about

    # Who is speaking
    section: str = ""                 # one of the constants above
    parties: list = field(default_factory=list)
    signatories: str = ""

    # Links to other data
    punkter: list = field(default_factory=list)   # join key to voting records
    number: int | None = None                     # reservation ordinal

    # Reservation numbers this chunk points at. Populated for decision and
    # deliberation chunks from the document's own cross-references, and used
    # to give deliberation chunks their punkter — they have no number of
    # their own, because one "förslag i korthet" box can cover many punkter.
    reservation_refs: list = field(default_factory=list)

    # Provenance
    dok_id: str = ""
    beteckning: str = ""
    rm: str = ""
    utskott: str = ""
    doc_title: str = ""

    @property
    def n_chars(self):
        return len(self.text)

    def context_line(self):
        """Prefix prepended before embedding, so that party and setting end up
        in the vector rather than only in the metadata."""
        parts = []
        if self.section == RESERVATION and self.number:
            parts.append(f"Reservation {self.number}")
        elif self.section == DISSENT and self.number:
            parts.append(f"Särskilt yttrande {self.number}")
        elif self.section in SECTION_LABELS:
            label = SECTION_LABELS[self.section]
            if self.punkter:
                punkter = ", ".join(str(p) for p in self.punkter)
                label = f"{label}, punkt {punkter}"
            parts.append(label)
        if self.parties:
            parts.append(", ".join(self.parties))
        if self.beteckning:
            parts.append(f"{self.beteckning} {self.rm}")
        if self.heading:
            parts.append(f"om {self.heading}")
        return ", ".join(parts) + ":"

    def for_embedding(self):
        return f"{self.context_line()}\n{self.text}"

    def to_dict(self):
        d = asdict(self)
        d["parties"] = ";".join(self.parties)
        d["punkter"] = ";".join(str(p) for p in self.punkter)
        d["reservation_refs"] = ";".join(str(r) for r in self.reservation_refs)
        d["n_chars"] = self.n_chars
        return d


# --------------------------------------------------------------------------
# Step 0: triage — is this file parseable at all?
# --------------------------------------------------------------------------

def read_html(path):
    """Read a file as text. Kept separate so callers can classify the raw
    HTML and parse it without opening the file twice."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def document_kind(raw_html):
    """Classify a file before parsing. Only 'word' is parseable.

    empty      a stub with no document in it (34)
    plaintext  the whole document inside one <pre> (1, KU10)
    pdfscan    PDF-to-HTML: positioned divs, no styles (13, mostly KU20)
    word       Word export with CSS classes — the real corpus (1434)
    """
    if len(raw_html) < MIN_DOCUMENT_BYTES:
        return "empty"
    if "<pre" in raw_html[:2000]:
        return "plaintext"
    if PDFSCAN_RE.search(raw_html[:4000]):
        return "pdfscan"
    return "word"


# --------------------------------------------------------------------------
# Step 1: read the file and split it into paragraphs
# --------------------------------------------------------------------------

def is_layout_table(table):
    """Word wraps reservation and dissent headings in small tables to hang
    the number in the margin. Appendix data tables start at 16 cells."""
    return len(table.find_all("td")) < DATA_TABLE_MIN_CELLS


def clean_text(element):
    """Normalise the text of a bs4 element.

    get_text() joins adjacent strings without a separator, which matters:
    Word splits single words across several <span> elements.
    """
    for br in element.find_all("br"):
        br.replace_with(" ")
    text = element.get_text()
    text = text.replace("\xad", "")     # soft hyphen from Word hyphenation
    text = text.replace("\xa0", " ")    # non-breaking space
    return re.sub(r"\s+", " ", text).strip()


def paragraphs_from_html(raw_html):
    """Return [(css_class, text), ...] in document order.

    Paragraphs inside data tables are skipped: they hold the appendix
    figures. Layout tables are kept — the reservation and dissent headings
    live inside them.
    """
    soup = BeautifulSoup(raw_html, "html.parser")

    paragraphs = []
    for p in soup.find_all("p"):
        table = p.find_parent("table")
        if table is not None and not is_layout_table(table):
            continue
        css_class = " ".join(p.get("class", []))
        text = clean_text(p)
        if text:
            paragraphs.append((css_class, text))
    return paragraphs


def read_paragraphs(path):
    """Convenience wrapper for callers that only have a path."""
    return paragraphs_from_html(read_html(path))


def is_toc(css_class):
    """Table-of-contents entries repeat every heading in the document and
    must never be mistaken for the heading itself."""
    return css_class.startswith("TOC") or css_class == "Innehllsfrteckning"


# --------------------------------------------------------------------------
# Step 2: document-level fields and section boundaries
# --------------------------------------------------------------------------

def document_info(path, paragraphs):
    """Return the fields that describe the whole document.

    rm and beteckning come from the Dokumentbeteckning paragraph, not from
    the filename or the CSV manifest — they are the join key to the voting
    records and must not depend on how the zip was unpacked. utskott still
    comes from the manifest.
    """
    info = {"dok_id": Path(path).stem.upper(), "doc_title": "",
            "rm": "", "beteckning": ""}
    for css_class, text in paragraphs:
        if css_class == DOC_TITLE_CLASS and not info["doc_title"]:
            info["doc_title"] = text
        elif css_class == DOKBET_CLASS and not info["beteckning"]:
            m = DOKBET_RE.match(text.strip())
            if m:
                info["rm"] = m.group("rm")
                info["beteckning"] = m.group("beteckning")
    return info


def split_sections(paragraphs):
    """Split the paragraph list into blocks, one per boundary heading.

    Returns [(css_class, heading, body), ...] where body is a list of
    (css_class, text) pairs. Everything from the first appendix is dropped.
    """
    blocks = []
    current = None

    for css_class, text in paragraphs:
        if css_class in STOP_AT:
            break
        if css_class in BOUNDARIES:
            if current:
                blocks.append(current)
            current = (css_class, text, [])
        elif current:
            current[2].append((css_class, text))

    if current:
        blocks.append(current)
    return blocks


def find_sections(paragraphs):
    """Locate the top-level sections by heading TEXT.

    Returns {heading_key: (start, end)} as indices into paragraphs, where
    start is the first paragraph AFTER the heading. Everything from the
    first appendix is excluded, and table-of-contents entries are ignored.
    """
    end = len(paragraphs)
    for i, (css_class, _) in enumerate(paragraphs):
        if css_class in STOP_AT:
            end = i
            break

    marks = []
    for i, (css_class, text) in enumerate(paragraphs[:end]):
        if is_toc(css_class):
            continue
        if text.strip().lower() in TOP_LEVEL_HEADINGS:
            marks.append((text.strip().lower(), i))

    sections = {}
    for n, (key, start) in enumerate(marks):
        stop = marks[n + 1][1] if n + 1 < len(marks) else end
        if key not in sections:          # first occurrence wins
            sections[key] = (start + 1, max(start + 1, stop))
    return sections


# --------------------------------------------------------------------------
# Step 3: reservations and dissents
# --------------------------------------------------------------------------

def parse_heading(text):
    """Split a reservation heading into topic, punkter and parties.

    "Riktlinjerna ..., punkt 1 (S)"  -> ("Riktlinjerna ...", [1], ["S"])
    "Statens budget ... (MP)"        -> ("Statens budget ...", [], ["MP"])
    "Vårändringsbudget ..., punkt 1 (S) – utrikes underrättelsetjänst"
        -> ("Vårändringsbudget ... – utrikes underrättelsetjänst", [1], ["S"])

    Returns None if the heading does not match.
    """
    m = HEADING_RE.match(text)
    if not m:
        return None

    punkter = [int(n) for n in re.findall(r"\d+", m.group("punkter") or "")]
    parties = [p.strip() for p in m.group("parties").split(",")]

    heading = m.group("heading").strip().rstrip(",")
    tail = (m.group("tail") or "").strip(" –—-")
    if tail:
        heading = f"{heading} – {tail}"

    return heading, punkter, parties


def extract_reservations(blocks, **doc_fields):
    """Turn reservation and dissent blocks into Chunk records.

    The ordinal comes from position: reservations are numbered in document
    order. The layout table also puts the *next* number at the end of the
    previous body, so trailing bare numbers are dropped.
    """
    chunks = []
    counters = {RESERVATION: 0, DISSENT: 0}

    for css_class, heading, body in blocks:
        section = SECTION_BY_CLASS.get(css_class)
        if section is None:
            continue

        parsed = parse_heading(heading)
        if parsed is None:
            continue
        topic, punkter, parties = parsed

        texts = [t for _, t in body]
        while texts and BARE_NUMBER_RE.match(texts[-1]):
            texts.pop()

        signatories = ""
        if texts and SIGNATORY_RE.match(texts[0]):
            signatories = texts.pop(0)

        counters[section] += 1
        chunks.append(Chunk(
            text=" ".join(texts),
            heading=topic,
            section=section,
            parties=parties,
            signatories=signatories,
            punkter=punkter,
            number=counters[section],
            **doc_fields,
        ))
    return chunks


# --------------------------------------------------------------------------
# Step 4: the majority side — what the committee decided and why
# --------------------------------------------------------------------------

def extract_decisions(paragraphs, span, **doc_fields):
    """One DECISION chunk per numbered punkt in the decision list.

    The list repeats: a bare number "1.", the topic title, the decision text
    ("Riksdagen avslår motionerna …"), then any reservations hung in the
    margin as "Reservation 1 (S)". Those margin references are an independent
    statement of which reservation belongs to which punkt, and agree with the
    reservation headings 99.1% of the time.
    """
    chunks = []
    number = None
    heading, body, refs = "", [], []
    expect_title = False

    def flush():
        if number is not None:
            chunks.append(Chunk(
                text=" ".join(body),
                heading=heading,
                section=DECISION,
                punkter=[number],
                number=number,
                reservation_refs=list(refs),
                **doc_fields,
            ))

    for css_class, text in paragraphs[span[0]:span[1]]:
        if css_class == KORTHET_CLASS:
            break       # överväganden has begun; the decision list is over
        if css_class in PUNKT_NUMBER_CLASSES and BARE_NUMBER_RE.match(text):
            flush()
            number = int(text[:-1])
            heading, body, refs = "", [], []
            expect_title = True
            continue
        if number is None:
            continue
        if expect_title:
            heading = text
            expect_title = False
            continue
        if css_class == RES_REF_CLASS:
            found = NUMBER_RE.findall(text)
            if found:
                refs.append(int(found[0]))
            continue
        if css_class in BLOCK_HEADING_CLASSES:
            continue
        body.append(text)

    flush()
    return chunks


def extract_deliberations(paragraphs, span, **doc_fields):
    """SUMMARY and DELIBERATION chunks from "Utskottets överväganden".

    The section repeats one block per topic, each opening with a "Utskottets
    förslag i korthet" box: a one-line statement of the decision, optionally
    a "Jämför reservation 1 (S), 2 (C)" cross-reference, then background, and
    finally "Utskottets ställningstagande" — the majority's own reasoning,
    which is the single most valuable text in the document for a neutral
    question.

    These blocks are NOT numbered and do NOT correspond one-to-one with the
    punkter: one box routinely covers many (25 punkter under 11 boxes is
    typical). Their punkter are filled in later by resolve_punkter().
    """
    chunks = []
    blocks = []
    current = None

    for css_class, text in paragraphs[span[0]:span[1]]:
        if (css_class == KORTHET_CLASS
                and text.strip().lower() == KORTHET_HEADING):
            if current:
                blocks.append(current)
            current = {"korthet": [], "refs": [], "standpoint": [],
                       "in_standpoint": False}
            continue
        if current is None:
            continue
        if css_class == KORTHET_BODY_CLASS:
            current["korthet"].append(text)
            continue
        if css_class == COMPARE_CLASS:
            current["refs"] = [int(n) for n in NUMBER_RE.findall(text)]
            continue
        if not is_toc(css_class) and text.strip().lower() == STANDPOINT_HEADING:
            current["in_standpoint"] = True
            continue
        if current["in_standpoint"]:
            if css_class in BLOCK_HEADING_CLASSES:
                current["in_standpoint"] = False    # a new block starts here
                continue
            current["standpoint"].append(text)

    if current:
        blocks.append(current)

    for block in blocks:
        korthet = " ".join(block["korthet"])
        if korthet:
            chunks.append(Chunk(
                text=korthet,
                section=SUMMARY,
                reservation_refs=list(block["refs"]),
                **doc_fields,
            ))
        if block["standpoint"]:
            chunks.append(Chunk(
                text=" ".join(block["standpoint"]),
                heading=korthet,        # the box is the block's only title
                section=DELIBERATION,
                reservation_refs=list(block["refs"]),
                **doc_fields,
            ))
    return chunks


def resolve_punkter(chunks):
    """Give the majority-side chunks their punkter, in place.

    A "förslag i korthet" box has no number of its own, but it names the
    reservations it is compared against, and those reservations state their
    punkt in their heading. Blocks with no "Jämför" line get no punkt — that
    means nobody reserved on the topic, so it was decided without a vote and
    there is nothing to link to.
    """
    by_number = {}
    for c in chunks:
        if c.section == RESERVATION and c.number:
            by_number[c.number] = c.punkter

    for c in chunks:
        if c.section in (SUMMARY, DELIBERATION) and c.reservation_refs:
            punkter = set()
            for ref in c.reservation_refs:
                punkter.update(by_number.get(ref, []))
            c.punkter = sorted(punkter)
    return chunks


def check_decision_refs(chunks):
    """Third checksum, independent of the summary sentence.

    The decision list hangs "Reservation 1 (S)" beside each punkt; the
    reservation heading states ", punkt 1". Two unrelated places in the
    document, so disagreement means one of them was misparsed.

    Returns (agreements, disagreements).
    """
    by_number = {c.number: c.punkter for c in chunks
                 if c.section == RESERVATION and c.number}
    agree = disagree = 0
    for c in chunks:
        if c.section != DECISION or not c.reservation_refs:
            continue
        if all(c.number in by_number.get(r, []) for r in c.reservation_refs):
            agree += 1
        else:
            disagree += 1
    return agree, disagree


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def summary_text(paragraphs):
    """Return the summary as one string, or "" if there is none.

    Two strategies, because 21 documents have no summary heading at all:
    the heading if there is one, otherwise everything between the document
    title and the first section heading. Located by heading TEXT rather than
    CSS class, because 92 documents style that heading as "R2".
    """
    for start_class in (None, DOC_TITLE_CLASS):
        collected, inside = [], False
        for css_class, text in paragraphs:
            key = text.strip().lower()
            if start_class is None:
                if key in SUMMARY_HEADINGS:
                    inside = True
                    continue
            elif css_class == start_class:
                inside = True
                continue
            if inside and key in SUMMARY_ENDS:
                break
            if inside and key in SUMMARY_HEADINGS:
                continue        # the fallback must not collect the heading
            if inside:
                collected.append(text)
        if collected:
            return " ".join(collected)
    return ""


def expected_counts(paragraphs):
    """What the summary claims, as {RESERVATION: n|None, DISSENT: n|None}.

    Every betänkande normally states the counts in plain Swedish, e.g.
    "I betänkandet finns åtta reservationer (S, V, C, MP)." That sentence is
    a free checksum for the whole corpus: no hand-written ground truth needed.

    Three states per kind, and the distinction is what makes the checksum
    cover the whole corpus rather than only the documents that happen to
    have reservations:

      n     the summary states this many
      0     the summary never mentions this kind, so there are none — a
            unanimous betänkande does not say "finns 0 reservationer"
      None  the summary mentions the kind without counting it, e.g. "I ett
            särskilt yttrande förklarar företrädare för …". Unverifiable,
            and must not be scored as if it claimed zero.
    """
    summary = summary_text(paragraphs)
    counts = {RESERVATION: None, DISSENT: None}
    if len(summary) < MIN_SUMMARY_CHARS:
        return counts

    for clause in CLAUSE_RE.finditer(summary):
        for m in KIND_RE.finditer(clause.group("body")):
            word = m.group("count").lower()
            count = NUMBER_WORDS.get(word)
            if count is None:
                count = int(word) if word.isdigit() else None
            if count is None:
                continue
            section = (RESERVATION if "reservation" in m.group("kind").lower()
                       else DISSENT)
            # First claim wins. The headline sentence comes first; a later
            # "I samma förslagspunkt finns en motivreservation" must not
            # overwrite "I betänkandet finns 13 reservationer".
            if counts[section] is None:
                counts[section] = count

    for section, pattern in KIND_WORDS.items():
        if counts[section] is None and not pattern.search(summary):
            counts[section] = 0
    return counts


def actual_counts(chunks):
    """Count distinct reservations and dissents in the parser output.

    Counts distinct numbers, not chunks, so that splitting a long
    reservation into several chunks later does not break the check.
    """
    return {
        RESERVATION: len({c.number for c in chunks if c.section == RESERVATION}),
        DISSENT: len({c.number for c in chunks if c.section == DISSENT}),
    }


def verify(expected, actual):
    """Compare a claim against the parser output.

    Returns "pass", "fail" or "unverifiable". A document where the summary
    counts one kind and is silent about the other is still checked on the
    kind it does count.
    """
    if expected[RESERVATION] is None and expected[DISSENT] is None:
        return "unverifiable"
    for section in (RESERVATION, DISSENT):
        claim = expected[section]
        if claim is None:
            continue            # silent about this kind; nothing to check
        if claim != actual[section]:
            return "fail"
    return "pass"


# Hand-read from the two reference documents. Kept as a regression test.
GROUND_TRUTH = {
    "HA01AU1": {
        RESERVATION: [],
        DISSENT: [
            {"number": 1, "parties": ["S"]},
            {"number": 2, "parties": ["V"]},
            {"number": 3, "parties": ["C"]},
            {"number": 4, "parties": ["MP"]},
        ],
    },
    "HA01FIU1": {
        RESERVATION: [
            {"number": 1, "parties": ["S"], "punkter": [1]},
            {"number": 2, "parties": ["V"], "punkter": [1]},
            {"number": 3, "parties": ["C"], "punkter": [1]},
            {"number": 4, "parties": ["MP"], "punkter": [1]},
            {"number": 5, "parties": ["S"], "punkter": [2]},
            {"number": 6, "parties": ["V"], "punkter": [2]},
            {"number": 7, "parties": ["C"], "punkter": [2]},
            {"number": 8, "parties": ["MP"], "punkter": [2]},
        ],
        DISSENT: [],
    },
}


def check(dok_id, chunks):
    """Compare parser output against the hand-read ground truth.

    Returns a list of mismatches; an empty list means the document parsed
    correctly. Only the two reference documents are covered — use
    expected_counts() for the rest of the corpus.
    """
    truth = GROUND_TRUTH.get(dok_id)
    if truth is None:
        return [f"no ground truth for {dok_id}"]

    problems = []
    for section in (RESERVATION, DISSENT):
        expected = truth[section]
        got = [c for c in chunks if c.section == section]

        got_numbers = sorted({c.number for c in got if c.number})
        expected_numbers = sorted(e["number"] for e in expected)
        if got_numbers != expected_numbers:
            problems.append(
                f"{section}: expected {expected_numbers}, got {got_numbers}"
            )
            continue

        for e in expected:
            match = next((c for c in got if c.number == e["number"]), None)
            if match is None:
                continue
            if match.parties != e["parties"]:
                problems.append(
                    f"{section} {e['number']}: expected parties "
                    f"{e['parties']}, got {match.parties}"
                )
            if match.punkter != e.get("punkter", []):
                problems.append(
                    f"{section} {e['number']}: expected punkter "
                    f"{e.get('punkter', [])}, got {match.punkter}"
                )
    return problems


def parse_document(path, raw_html=None):
    """Full pipeline for one file. Returns (info, paragraphs, blocks, chunks).

    chunks holds every section: reservations and dissents from the minority
    side, and decisions, korthet summaries and deliberations from the
    majority side.

    paragraphs is returned because expected_counts() needs it: in 92 documents
    the summary heading is not a boundary class, so it never becomes a block.

    Pass raw_html if you have already read the file (to classify it) and want
    to avoid a second read.
    """
    if raw_html is None:
        raw_html = read_html(path)
    paragraphs = paragraphs_from_html(raw_html)
    info = document_info(path, paragraphs)
    blocks = split_sections(paragraphs)

    chunks = extract_reservations(blocks, **info)

    sections = find_sections(paragraphs)
    if DECISION_HEADING in sections:
        chunks += extract_decisions(paragraphs, sections[DECISION_HEADING],
                                    **info)
    if DELIBERATION_HEADING in sections:
        chunks += extract_deliberations(
            paragraphs, sections[DELIBERATION_HEADING], **info)

    resolve_punkter(chunks)
    return info, paragraphs, blocks, chunks


if __name__ == "__main__":
    import sys

    for path in sys.argv[1:]:
        raw = read_html(path)
        kind = document_kind(raw)
        if kind != "word":
            print(f"{Path(path).stem.upper():10s} SKIPPED ({kind})")
            continue

        info, paragraphs, blocks, chunks = parse_document(path, raw)
        problems = check(info["dok_id"], chunks)
        status = "OK" if not problems else f"{len(problems)} MISMATCHES"
        verdict = verify(expected_counts(paragraphs), actual_counts(chunks))
        agree, disagree = check_decision_refs(chunks)
        counts = {}
        for c in chunks:
            counts[c.section] = counts.get(c.section, 0) + 1
        print(f"{info['dok_id']:10s} {len(chunks):3d} chunks {counts}  "
              f"{status}  checksum:{verdict}  refs:{agree}/{agree + disagree}")
        for p in problems:
            print(f"    {p}")