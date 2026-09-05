import re
from dataclasses import dataclass, field, asdict
from bs4 import BeautifulSoup
from pathlib import Path


SUMMARY = "summary"
DECISION = "decision"
DELIBERATION = "deliberation"
RESERVATION = "reservation"
DISSENT = "dissent"


BOUNDARIES = {
    "Sammanfattning",
    "Avsnittsrubrik",
    "Reservationsrubrik",
    "Srskiltyttranderubrik",
}

@dataclass
class Chunk:
    #One unit in the vector index
    
    text: str = ""      #what gets embedded
    heading: str = ""   #what the chunk is about
    
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
 
#Step 1: read the file and split it into paragraphs


DATA_TABLE_MIN_CELLS = 8


def is_layout_table(table):
    """Word wraps reservation and dissent headings in small tables to hang
    the number in the margin. Appendix data tables start at 16 cells.
    """
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
    figures. Layout tables are kept — the reservation and dissent
    headings live inside them.
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


def document_info(path, paragraphs):
    """Return the fields that describe the whole document."""
    dok_id = Path(path).stem.upper()
    title = ""
    for css_class, text in paragraphs:
        if css_class == "DokumentRubrik":
            title = text
            break
    return {"dok_id": dok_id, "doc_title": title}

STOP_AT = {"Bilaga", "Bilagerubrik"}

def split_sections(paragraphs):
    """Split the paragraph list into blocks, one per boundary heading.

    Returns [(css_class, heading, body), ...] where body is a list of
    (css_class, text) pairs. Everything from the first appendix is dropped.
    """
    blocks = []
    current = None

    for css_class, text in paragraphs:
        if css_class == STOP_AT:
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

PARTY = r"[A-ZÅÄÖ]{1,3}"

HEADING_RE = re.compile(
    rf"^(?P<heading>.+?)"
    rf"(?:,\s*punkt(?:erna)?\s+(?P<punkter>\d+(?:\s*(?:,|och)\s*\d+)*))?"
    rf"\s*\((?P<parties>{PARTY}(?:\s*,\s*{PARTY})*)\)$"
)

BARE_NUMBER_RE = re.compile(r"^\d+\.$")

SECTION_BY_CLASS = {
    "Reservationsrubrik": RESERVATION,
    "Srskiltyttranderubrik": DISSENT,
}


def parse_heading(text):
    """Split a reservation heading into topic, punkter and parties.

    "Riktlinjerna ..., punkt 1 (S)"  ->  ("Riktlinjerna ...", [1], ["S"])
    "Statens budget ... (MP)"        ->  ("Statens budget ...", [], ["MP"])
    Returns None if the heading does not match.
    """
    m = HEADING_RE.match(text)
    if not m:
        return None
    punkter = [int(n) for n in re.findall(r"\d+", m.group("punkter") or "")]
    parties = [p.strip() for p in m.group("parties").split(",")]
    return m.group("heading").strip().rstrip(","), punkter, parties


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
        if texts and texts[0].startswith("av "):
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