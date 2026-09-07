"""Parser for Swedish parliamentary committee reports (betänkanden).

Turns riksdagen's HTML export into Chunk records ready for embedding.

GLOSSARY — Swedish domain terms kept untranslated on purpose, because they
name institutional concepts and must match the source data and riksdagen's
own field names:

  betänkande   committee report; the document type we parse
  reservation  formal dissent by one or more parties; goes to a vote
  yttrande     "särskilt yttrande", a weaker dissent that is NOT voted on
  punkt        numbered decision item; the join key to voting records
  beteckning   report designation, e.g. "FiU1"
  rm           "riksmöte", the parliamentary session, e.g. "2022/23"
  dok_id       riksdagen's document identifier, e.g. "HA01FIU1"
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

# "I betänkandet finns åtta reservationer (S, V, C, MP)."
EXPECTED_RE = re.compile(
    r"finns\s+(?P<count>\w+)\s+"
    r"(?P<kind>reservationer?|särskilda\s+yttranden|särskilt\s+yttrande)",
    re.IGNORECASE,
)

NUMBER_WORDS = {
    "en": 1, "ett": 1, "två": 2, "tre": 3, "fyra": 4, "fem": 5,
    "sex": 6, "sju": 7, "åtta": 8, "nio": 9, "tio": 10, "elva": 11,
    "tolv": 12, "tretton": 13, "fjorton": 14, "femton": 15,
    "sexton": 16, "sjutton": 17, "arton": 18, "nitton": 19, "tjugo": 20,
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
        d["n_chars"] = self.n_chars
        return d


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


def read_paragraphs(path):
    """Return [(css_class, text), ...] in document order.

    Paragraphs inside data tables are skipped: they hold the appendix
    figures. Layout tables are kept — the reservation and dissent headings
    live inside them.
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        soup = BeautifulSoup(fh.read(), "html.parser")

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


# --------------------------------------------------------------------------
# Step 2: document-level fields and section boundaries
# --------------------------------------------------------------------------

def document_info(path, paragraphs):
    """Return the fields that describe the whole document.

    beteckning, rm and utskott are not in the HTML body; they are joined in
    later from the CSV manifest, which already has them keyed by dok_id.
    """
    dok_id = Path(path).stem.upper()
    title = ""
    for css_class, text in paragraphs:
        if css_class == "DokumentRubrik":
            title = text
            break
    return {"dok_id": dok_id, "doc_title": title}


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
# Validation
# --------------------------------------------------------------------------

def expected_counts(blocks):
    """Read the reservation and dissent counts stated in the summary.

    Every betänkande states them in plain Swedish, e.g. "I betänkandet finns
    åtta reservationer (S, V, C, MP)." That sentence is a free checksum for
    the whole corpus: no hand-written ground truth needed.

    Returns {RESERVATION: int, DISSENT: int}; a missing statement means zero.
    """
    summary = ""
    for css_class, heading, body in blocks:
        if css_class == "Sammanfattning":
            summary = " ".join(t for _, t in body)
            break

    counts = {RESERVATION: 0, DISSENT: 0}
    for m in EXPECTED_RE.finditer(summary):
        word = m.group("count").lower()
        count = NUMBER_WORDS.get(word)
        if count is None:
            if not word.isdigit():
                continue
            count = int(word)
        if m.group("kind").lower().startswith("reservation"):
            counts[RESERVATION] = count
        else:
            counts[DISSENT] = count
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


def parse_document(path):
    """Full pipeline for one file. Returns (info, blocks, chunks)."""
    paragraphs = read_paragraphs(path)
    info = document_info(path, paragraphs)
    blocks = split_sections(paragraphs)
    chunks = extract_reservations(blocks, **info)
    return info, blocks, chunks


if __name__ == "__main__":
    import sys

    for path in sys.argv[1:]:
        info, blocks, chunks = parse_document(path)
        problems = check(info["dok_id"], chunks)
        status = "OK" if not problems else f"{len(problems)} MISMATCHES"
        print(f"{info['dok_id']:10s} {len(chunks):3d} chunks  {status}")
        for p in problems:
            print(f"    {p}")