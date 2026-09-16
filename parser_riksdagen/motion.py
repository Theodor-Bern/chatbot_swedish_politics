"""Parser for Swedish parliamentary motions (motioner) — first pass.

En motion har två delar som vi bryr oss om:

  yrkande     ett enskilt, formellt beslutsförslag: "Riksdagen ställer sig
              bakom ... och tillkännager detta för regeringen." Ligger som
              <li class="Frslagstext"> i listan under "Förslag till
              riksdagsbeslut".
  sektion     rubrik + brödtext under den (numrerade Rubrik1/2/3numrerat).
              Innehåller argumenten, och ofta en nästan ordagrant likalydande
              omskrivning av ett yrkande i sin sista mening.

Det finns ingen explicit koppling i HTML:en mellan ett yrkande och sin
sektion — bara att texten återupprepas. Vi hittar kopplingen genom
ordöverlapp: vilken sektion delar flest ovanliga ord med yrkandet.

Kör:
    python -m parser_riksdagen.motion data/motioner/2022_23_1.html
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

# .Frslagstext, .Hemstlatt och .Yrkande delar bara CSS-formatering i Aspose-
# exporten (se <style>-blocket) — inget som säger att de är samma sak
# semantiskt, men i praktiken markerar alla tre en yrkanderad.
YRKANDE_CLASSES = {"Frslagstext", "Hemstlatt", "Yrkande"}
RUBRIK_CLASSES = {"Rubrik1numrerat", "Rubrik2numrerat", "Rubrik3numrerat", "Rubrik4numrerat"}

BETECKNING_RE = re.compile(r"^(?P<rm>\d{4}/\d{2}):(?P<nr>\S+)")
PARTY_RE = re.compile(r"\(([A-ZÅÄÖ]{1,3})\)\s*$")

WORD_RE = re.compile(r"[a-zåäöA-ZÅÄÖ0-9]+")
STOPP = set("""och att det som en för av på är den till med de i om så har inte
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
# dokumentmetadata
# --------------------------------------------------------------------------

def document_info(soup):
    """rm, beteckning, parti, förste undertecknare, titel."""
    info = {"rm": "", "beteckning": "", "parti": "", "forste_undertecknare": "",
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
        info["forste_undertecknare"] = author_text
        m = PARTY_RE.search(author_text)
        if m:
            info["parti"] = m.group(1)

    title = soup.find("h1")
    if title:
        info["doc_title"] = clean_text(title)

    return info


def extract_signatories(soup):
    """Namn på alla undertecknare, från underskriftstabellen längst ner."""
    namn = []
    for p in soup.find_all("p", class_="Underskrifter"):
        text = clean_text(p)
        if text:
            namn.append(text)
    return namn


# --------------------------------------------------------------------------
# yrkanden
# --------------------------------------------------------------------------

def extract_yrkanden(soup):
    """En sträng per <li> i "Förslag till riksdagsbeslut"-listan, i ordning."""
    yrkanden = []
    for li in soup.find_all("li"):
        if set(li.get("class", []) or []) & YRKANDE_CLASSES:
            text = clean_text(li)
            if text:
                yrkanden.append(text)
    return yrkanden


# --------------------------------------------------------------------------
# rubriksektioner
# --------------------------------------------------------------------------

def extract_sections(soup):
    """[{'heading': str, 'body': [str, ...]}, ...] — en post per numrerad
    rubrik, med all brödtext fram till nästa rubrik.

    <li> hoppas över: de hör till yrkande-listan, inte till en sektions
    brödtext, även om de råkar ligga inuti Section1.
    """
    section1 = soup.find("div", class_="Section1")
    if section1 is None:
        return []

    sections = []
    current = None
    for p in section1.find_all("p", recursive=True):
        if p.find_parent("li"):
            continue
        classes = set(p.get("class", []) or [])
        text = clean_text(p)
        if not text:
            continue
        if classes & RUBRIK_CLASSES:
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
# matchning: vilket yrkande hör till vilken sektion?
# --------------------------------------------------------------------------

def tokenize(text):
    words = (w.lower() for w in WORD_RE.findall(text))
    return {w for w in words if w not in STOPP and len(w) > 2}


def best_match(yrkande, sections):
    """Sektionen (index) vars brödtext delar störst andel av yrkandets
    ovanliga ord, och den andelen. (None, 0.0) om inget delar något alls.
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
# CLI: kör och skriv ut vad vi hittar
# --------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)

    path = Path(sys.argv[1])
    soup = BeautifulSoup(read_html(path), "html.parser")

    info = document_info(soup)
    signatories = extract_signatories(soup)
    yrkanden = extract_yrkanden(soup)
    sections = extract_sections(soup)

    print(f"dok:      {info['beteckning']} ({info['rm']})")
    print(f"parti:    {info['parti']}")
    print(f"titel:    {info['doc_title']}")
    print(f"1:a namn: {info['forste_undertecknare']}")
    print(f"alla:     {', '.join(signatories)}")
    print(f"\n{len(yrkanden)} yrkanden, {len(sections)} sektioner\n")

    for n, yrkande in enumerate(yrkanden, 1):
        i, score = best_match(yrkande, sections)
        match = f"[{score:.0%}] {sections[i]['heading']}" if i is not None else "INGEN MATCH"
        print(f"{n}. {yrkande[:90]}...")
        print(f"   -> {match}\n")


if __name__ == "__main__":
    main()
