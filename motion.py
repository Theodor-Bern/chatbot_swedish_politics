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
    python -m parser_riksdagen.motion inspect data/motioner/2022_23_1.html
    python -m parser_riksdagen.motion build data/motioner out/motion_chunks.jsonl
    python -m parser_riksdagen.motion embed out/motion_chunks.jsonl out/motion_index
    python -m parser_riksdagen.motion embed out/motion_chunks.jsonl out/motion_index --stub
    python -m parser_riksdagen.motion search out/motion_index "vad vill V göra åt kärnkraft?"
    python -m parser_riksdagen.motion shell out/motion_index
    python -m parser_riksdagen.motion fraga out/motion_index "vad vill V göra åt kärnkraft?"
    python -m parser_riksdagen.motion fraga out/motion_index "..." --torrkor
    python -m parser_riksdagen.motion chatt out/motion_index

Kräver en nyckel från Google AI Studio för fraga/chatt (inte för --torrkor):
    export GEMINI_API_KEY=...        # lägg ALDRIG nyckeln i repot
"""

from __future__ import annotations

import json
import math
import pickle
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup

# Under denna gräns litar vi inte på ordöverlapps-matchningen — bättre att
# spara yrkandet utan sektion än att gissa fel. Satt löst utifrån två
# motioner (rätta matchningar låg på 85-96%, en observerad felmatchning på
# 33%) — höj/sänk när fler motioner har testats.
MIN_MATCH = 0.5

# .Frslagstext, .Hemstlatt och .Yrkande delar bara CSS-formatering i Aspose-
# exporten (se <style>-blocket) — inget som säger att de är samma sak
# semantiskt, men i praktiken markerar alla tre en yrkanderad.
YRKANDE_CLASSES = {"Frslagstext", "Hemstlatt", "Yrkande"}
RUBRIK_CLASSES = {"Rubrik1numrerat", "Rubrik2numrerat", "Rubrik3numrerat", "Rubrik4numrerat"}
# Nyare exporter (med -aw-sdt-tag-omslutna divar) sätter riktiga <h1>-<h6>
# i stället för <p class="RubrikXnumrerat"> — samma roll, annan markup.
HEADING_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6"]

PARTY_NAMES = {
    "S": "Socialdemokraterna", "M": "Moderaterna", "SD": "Sverigedemokraterna",
    "C": "Centerpartiet", "V": "Vänsterpartiet", "KD": "Kristdemokraterna",
    "MP": "Miljöpartiet", "L": "Liberalerna",
}

# Parameterutvinning: fritext -> partikod. Samma tabell och princip som
# answer.py i huvudrepot — deterministisk uppslagning i stället för att
# hoppas att embeddingen/BM25 gissar rätt parti från frågetexten. Vi har
# tre gånger sett att den gissningen är opålitlig (kärnkraft->M,
# invandring->SD, skola->V/C, trots frågor riktade mot ett annat parti).
ALIAS = {
    "s": "S", "socialdemokraterna": "S", "socialdemokraterna s": "S",
    "sossarna": "S", "socialdemokratiska arbetarepartiet": "S",
    "m": "M", "moderaterna": "M", "moderata samlingspartiet": "M",
    "sd": "SD", "sverigedemokraterna": "SD",
    "c": "C", "centerpartiet": "C", "centern": "C",
    "v": "V", "vänsterpartiet": "V", "vansterpartiet": "V",
    "kd": "KD", "kristdemokraterna": "KD",
    "mp": "MP", "miljöpartiet": "MP", "miljopartiet": "MP",
    "de gröna": "MP", "l": "L", "liberalerna": "L", "folkpartiet": "L",
}


def parti_i_fragan(fråga):
    """Vilka partier nämns i frågan? Tom lista = inget nämnt (sök brett)."""
    text = f" {fråga.lower()} "
    funna = []
    for alias in sorted(ALIAS, key=len, reverse=True):
        möns = r"\b" + re.escape(alias) + r"\b"
        if re.search(möns, text):
            kod = ALIAS[alias]
            if kod not in funna:
                funna.append(kod)
    return funna


MODEL_NAME = "intfloat/multilingual-e5-large"
EMBED_BATCH = 32

BETECKNING_RE = re.compile(r"^(?P<rm>\d{4}/\d{2}):(?P<nr>\S+)")
# "(V)" — vanligast. "(båda C)"/"(alla S)" — när flera medundertecknare
# delar samma parti skriver Riksdagen ut det i klartext före partikoden.
PARTY_RE = re.compile(r"\((?:[a-zåäö]+\s+)?([A-ZÅÄÖ]{1,3})\)\s*$")

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
    """En sträng per yrkande, i ordning.

    Yrkandet ligger som <li class="Frslagstext"> i äldre exporter (en
    <ol>-lista under "Förslag till riksdagsbeslut"), men som ett fristående
    <p class="Frslagstext"> i nyare, -aw-sdt-tag-baserade exporter — ingen
    <ol>/<li> alls, bara en <div> med -aw-sdt-title "Yrkande N" runt ett
    vanligt stycke. Samma klass, olika tagg.
    """
    yrkanden = []
    for el in soup.find_all(["li", "p"]):
        if set(el.get("class", []) or []) & YRKANDE_CLASSES:
            text = clean_text(el)
            if text:
                yrkanden.append(text)
    return yrkanden


# --------------------------------------------------------------------------
# rubriksektioner
# --------------------------------------------------------------------------

def extract_sections(soup):
    """[{'heading': str, 'body': [str, ...]}, ...] — en post per rubrik,
    med all brödtext fram till nästa rubrik.

    En rubrik är antingen en <p class="RubrikXnumrerat"> (äldre exporter)
    eller en riktig <h1>-<h6> (nyare, -aw-sdt-tag-baserade exporter). Båda
    markerar samma sak — var brödtexten hör hemma — bara med olika markup.

    <li> hoppas över: de hör till yrkande-listan, inte till en sektions
    brödtext, även om de råkar ligga inuti Section1.
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
            continue    # ett fristående <p class="Frslagstext">, inte brödtext
        text = clean_text(el)
        if not text:
            continue
        is_rubrik = el.name in HEADING_TAGS or bool(classes & RUBRIK_CLASSES)
        if is_rubrik:
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

# Lätt svensk stamning (samma lista som rag_index.py) — klipper böjnings-
# ändelser så att "landsbygden"/"landsbygd" räknas som samma ord. Ordnad
# längst-först, och kapar bara om minst 4 tecken blir kvar, så korta ord
# som "det"/"stat" lämnas i fred.
SUFFIX = ["ernas", "arnas", "ornas", "andet", "arna", "erna", "orna",
          "ande", "ende", "aste", "ades", "ade", "are", "ast", "ens",
          "ets", "er", "ar", "or", "en", "et", "na", "as", "es"]
MIN_STAM = 4


def stam(ord_):
    for suf in SUFFIX:
        if ord_.endswith(suf) and len(ord_) - len(suf) >= MIN_STAM:
            return ord_[: -len(suf)]
    return ord_


def tokenize_list(text):
    """Som tokenize(), men en lista — BM25 behöver antal förekomster per
    ord (termfrekvens), inte bara vilka ord som finns."""
    words = (w.lower() for w in WORD_RE.findall(text))
    return [stam(w) for w in words if w not in STOPP and len(w) > 2]


def tokenize(text):
    return set(tokenize_list(text))


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
# BM25 — exakt ordmatchning, kompletterar vektorsökningen
# --------------------------------------------------------------------------

class BM25:
    """Standard BM25 med inverterat index. Ren stdlib, samma implementation
    som rag_index.py använder för betänkanden — se den för mer utförliga
    kommentarer om varje del av formeln."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.längder = [len(d) for d in docs]
        self.medel = sum(self.längder) / self.N if self.N else 0.0
        self.inv = defaultdict(list)          # term -> [(doc_idx, tf), ...]
        df = Counter()
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                self.inv[term].append((i, tf))
                df[term] += 1
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()
        }

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump({"k1": self.k1, "b": self.b, "N": self.N,
                         "längder": self.längder, "medel": self.medel,
                         "inv": dict(self.inv), "idf": self.idf},
                        fh, protocol=4)

    @classmethod
    def load(cls, path):
        d = pickle.load(open(path, "rb"))
        o = cls.__new__(cls)
        o.k1, o.b, o.N = d["k1"], d["b"], d["N"]
        o.längder, o.medel = d["längder"], d["medel"]
        o.inv, o.idf = d["inv"], d["idf"]
        return o

    def search(self, query, limit=300):
        poäng = defaultdict(float)
        for term in tokenize_list(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.inv.get(term, ()):
                norm = 1 - self.b + self.b * self.längder[i] / self.medel
                poäng[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return sorted(poäng.items(), key=lambda kv: -kv[1])[:limit]


# --------------------------------------------------------------------------
# chunk — det som faktiskt sparas
# --------------------------------------------------------------------------

@dataclass
class Chunk:
    """En chunk = ett yrkande, med sin matchade sektion som kontext."""

    text: str = ""                # yrkandet, helt och ofiltrerat — det som embeddas
    heading: str = ""              # rubriken på den matchade sektionen
    sektion_text: str = ""         # sektionens brödtext — kontext, inte embeddat direkt
    match_score: float = 0.0       # ordöverlappet som avgjorde matchningen (se MIN_MATCH)
    yrkande_nr: int = 0

    parti: str = ""
    rm: str = ""
    beteckning: str = ""
    doc_title: str = ""
    forste_undertecknare: str = ""
    undertecknare: list = field(default_factory=list)
    layer: str = "motion"          # eget lager — varken SAID eller DID, se resonemanget i chatten

    @property
    def chunk_id(self):
        return f"{self.rm}:{self.beteckning}:yrkande:{self.yrkande_nr}"

    def to_dict(self):
        d = asdict(self)
        d["chunk_id"] = self.chunk_id
        d["undertecknare"] = ";".join(self.undertecknare)
        return d


# Kända yrkande-mallar, isolerade så vi kan klippa bort dem ur det som
# embeddas (INTE ur chunk.text — LLM:en ska fortfarande få hela, korrekta
# meningen). Fler mallar läggs till här när vi hittar dem i riktiga
# chunkar. Matchar ingen av dem -> hela texten används oförändrad. Hellre
# det än att gissa och klippa fel.
YRKANDE_WRAPPER_PATTERNS = [
    re.compile(
        r"^Riksdagen ställer sig bakom det som anförs i motionen om (att )?"
        r"(?P<kärna>.+?)"
        r"[,.]?\s*och\s+(detta\s+tillkännager\s+riksdagen|tillkännager\s+detta)\s+för\s+regeringen\.?\s*$",
        re.IGNORECASE),
    # Avslagsyrkanden: "avseende X." eller "i den del som avser X." — ingen
    # avslutande "och tillkännager"-svans, meningen tar bara slut efter X.
    re.compile(
        r"^Riksdagen avslår regeringens förslag\s+"
        r"(avseende|i\s+den\s+del\s+som\s+avser)\s+"
        r"(?P<kärna>.+?)\.?\s*$",
        re.IGNORECASE),
]


def yrkande_kärna(text):
    for pat in YRKANDE_WRAPPER_PATTERNS:
        m = pat.match(text)
        if m:
            return m.group("kärna").strip()
    return text


def passage_core(chunk):
    """Kontextrad + yrkandets kärna — det gemensamma innehållet som både
    BM25 och embedding-modellen indexerar, bara paketerat olika (se
    passage_text() och BM25 nedan). Kontextraden sätter parti och ämne
    FÖRE texten (samma "wordalisation"-princip som rag_index.py använder
    för betänkanden); yrkande_kärna() klipper bort mallfrasen ("Riksdagen
    ställer sig bakom det som anförs i motionen om ... och tillkännager
    detta för regeringen") som annars konkurrerar om utrymme med
    tiotusentals andra chunkar.
    """
    namn = PARTY_NAMES.get(chunk.parti, chunk.parti)
    kontext = f"{namn} ({chunk.parti}), motion {chunk.beteckning} {chunk.rm}"
    if chunk.heading:
        kontext += f", om {chunk.heading}"
    return f"{kontext}\n{yrkande_kärna(chunk.text)}"


def passage_text(chunk):
    """Det som faktiskt embeddas: samma kärna, med e5:s "passage: "-prefix."""
    return f"passage: {passage_core(chunk)}"


def parse_motion(path):
    """En HTML-fil -> lista av Chunk, ett per yrkande i dokumentet."""
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
            parti=info["parti"],
            rm=info["rm"],
            beteckning=info["beteckning"],
            doc_title=info["doc_title"],
            forste_undertecknare=info["forste_undertecknare"],
            undertecknare=signatories,
        ))
    return chunks


# --------------------------------------------------------------------------
# build — kör mot en hel mapp av motioner
# --------------------------------------------------------------------------

def build(html_folder, out_path):
    # rglob, inte glob: bulk-exporten kan ligga en mapp per dokument
    # (t.ex. mot-2022-2025.html/hb01fiu1/....html), inte platt i toppnivån.
    files = sorted(Path(html_folder).rglob("*.html"))
    print(f"hittade {len(files)} .html-filer under {html_folder}", flush=True)
    if not files:
        sys.exit(f"hittar inga .html-filer i {html_folder} (även i undermappar)")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    osaker = 0
    crashed = []
    per_parti = Counter()

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
                per_parti[c.parti] += 1
                if c.match_score < MIN_MATCH:
                    osaker += 1

    print(f"läste {len(files)} filer, skrev {written} chunkar till {out_path}")
    print("  per parti: " + "  ".join(f"{p} {n}" for p, n in per_parti.most_common()))
    print(f"  utan säker sektionsmatchning (< {MIN_MATCH:.0%}): {osaker}/{written}")
    if crashed:
        print(f"  KRASCHADE ({len(crashed)}): {crashed}")


# --------------------------------------------------------------------------
# embed — chunkar -> vektorer
# --------------------------------------------------------------------------

def stub_vectors(texts, dim=256):
    """Deterministisk fejk-encoder för rökttest — INTE semantisk, bara ett
    sätt att bevisa att hela kedjan (läsa, bygga passage, spara vektorer)
    fungerar innan man väntar på att en 2 GB-modell laddas ner."""
    import numpy as np
    v = np.zeros((len(texts), dim), dtype="float32")
    for i, t in enumerate(texts):
        for tok in tokenize(t):
            v[i, hash(tok) % dim] += 1.0
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def encode(texts, stub=False, model_name=MODEL_NAME):
    if stub:
        return stub_vectors(texts)
    from sentence_transformers import SentenceTransformer
    print(f"laddar {model_name} …")
    modell = SentenceTransformer(model_name)
    modell.max_seq_length = 384
    return modell.encode(
        texts, batch_size=EMBED_BATCH, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    ).astype("float32")


def embed(chunks_path, out_dir, stub=False, model_name=MODEL_NAME):
    """Läs motion_chunks.jsonl, bygg en passage per chunk, embedda OCH
    BM25-indexera, spara.

    Skriver fyra filer i out_dir:
      vectors.npy   en rad per chunk, samma ordning som meta.jsonl
      bm25.pkl      det inverterade BM25-indexet över samma passages
      meta.jsonl    samma chunkar som lästes in — det man slår upp träffar i
      info.json     vilken modell, hur många chunkar, för att inte blanda
                    ihop index byggda med olika modeller av misstag
    """
    import numpy as np

    chunks_path = Path(chunks_path)
    rows = [json.loads(line) for line in open(chunks_path, encoding="utf-8")]
    print(f"{len(rows)} chunkar inlästa från {chunks_path}")

    chunkar = [Chunk(**{k: v for k, v in r.items()
                        if k in Chunk.__dataclass_fields__})
               for r in rows]
    cores = [passage_core(c) for c in chunkar]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "meta.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("bygger BM25 …")
    BM25([tokenize_list(c) for c in cores]).save(out_dir / "bm25.pkl")

    vek = encode([f"passage: {c}" for c in cores], stub=stub, model_name=model_name)
    np.save(out_dir / "vectors.npy", vek)

    with open(out_dir / "info.json", "w", encoding="utf-8") as fh:
        json.dump({"model": "stub" if stub else model_name,
                   "dim": int(vek.shape[1]), "n": len(rows)},
                  fh, ensure_ascii=False, indent=2)

    print(f"skrev {out_dir}/  ({vek.shape[0]} × {vek.shape[1]})")


# --------------------------------------------------------------------------
# search — fråga -> mest lika chunkar
# --------------------------------------------------------------------------

# RRF väger normalt de två listorna lika. rag_index.py:s utvärdering visar
# att BM25 är klart svagare än vektorlistan på naturligt formulerade
# frågor, men bra på att fånga upp exakta ord (som "kärnkraft" mot
# "kärnvapen") som vektorsökningen blandar ihop. Samma vikter som
# rag_index.py, som utgångspunkt — ingen egen utvärdering gjord här än.
RRF_K = 60
RRF_VIKT_BM25 = 0.35
RRF_VIKT_VEKTOR = 1.0


class Index:
    """Håller vektorerna och BM25-indexet i minnet. Använd 'shell' för
    flera frågor i rad — annars laddas embeddingmodellen om vid varje
    CLI-anrop."""

    def __init__(self, index_dir):
        import numpy as np
        index_dir = Path(index_dir)
        self.info = json.load(open(index_dir / "info.json", encoding="utf-8"))
        self.meta = [json.loads(l) for l in
                     open(index_dir / "meta.jsonl", encoding="utf-8")]
        self.vek = np.load(index_dir / "vectors.npy")
        self.bm = BM25.load(index_dir / "bm25.pkl")
        self.stub = self.info["model"] == "stub"

    def _encode_query(self, fråga):
        q = f"query: {fråga}"
        if self.stub:
            return stub_vectors([q])[0]
        if not hasattr(self, "_modell"):
            from sentence_transformers import SentenceTransformer
            self._modell = SentenceTransformer(self.info["model"])
        return self._modell.encode([q], normalize_embeddings=True,
                                   convert_to_numpy=True).astype("float32")[0]

    def search(self, fråga, k=8, parti=None, kandidater=300):
        import numpy as np

        bm_lista = self.bm.search(fråga, limit=kandidater)

        qv = self._encode_query(fråga)
        sim = self.vek @ qv        # kosinuslikhet, vektorerna är redan normaliserade
        vek_lista = [(int(i), float(sim[i])) for i in np.argsort(-sim)[:kandidater]]

        # RRF: rangordning slås ihop utan att normalisera två olika
        # poängskalor (BM25:s och kosinuslikhetens) mot varandra.
        rrf = defaultdict(float)
        for vikt, lista in ((RRF_VIKT_BM25, bm_lista), (RRF_VIKT_VEKTOR, vek_lista)):
            for rang, (i, _) in enumerate(lista):
                rrf[i] += vikt / (RRF_K + rang + 1)

        träffar = []
        for i, poäng in sorted(rrf.items(), key=lambda kv: -kv[1]):
            m = self.meta[i]
            if parti and m.get("parti") not in parti:
                continue
            träffar.append((poäng, m))
            if len(träffar) >= k:
                break
        return träffar


def visa(träffar):
    if not träffar:
        print("inga träffar")
        return
    for n, (poäng, m) in enumerate(träffar, 1):
        print(f"\n{n:2}. [{poäng:.3f}] {m['parti']} — motion {m['beteckning']} {m['rm']}")
        if m.get("heading"):
            print(f"    om: {m['heading']}")
        print(f"    {m['text']}")


# --------------------------------------------------------------------------
# svar — hämtning + LLM. Samma mönster som answer.py i huvudrepot.
# --------------------------------------------------------------------------

GEMINI_MODELL = "gemini-3.6-flash"
MAX_KONTEXT_TECKEN = 24000
MAX_UT_TOKENS = 4096

# Steg 1 och 4 (vem den är, hur den ska svara). Steg 2 (vad den vet) ligger
# i KUNSKAP nedan.
SYSTEM = """Du är en granskare av svensk partipolitik. Materialet du får är
motioner — formella förslag som riksdagsledamöter lämnar in i sitt partis
namn.

REGLER, i fallande ordning:

1. Du använder ENDAST det material du får i KONTEXT. Har du egna minnen av
   svensk politik är de föråldrade och du använder dem inte. Räcker
   kontexten inte för att svara säger du det rent ut i stället för att
   gissa eller koppla ihop lösa trådar.

2. Varje sakpåstående följs av sin källa i hakparentes, t.ex. [2] eller
   [motion 2751 2023/24]. Ett påstående utan källa får inte skrivas.

3. En motion är ETT FÖRSLAG partiet lämnat in — inte ett beslut, inte en
   lag, inte en bekräftad utfall. Skriv alltid "X har föreslagit" eller
   "X vill", ALDRIG "X har genomfört" eller "X har åstadkommit", eftersom
   materialet inte visar vad som hände sen (avslogs, bifölls, kom aldrig
   upp till omröstning). Nämner samma parti samma krav i flera motioner
   över flera år, konstatera det (partiet har återkommit till frågan) men
   gissa inte på VARFÖR — det vet du inte utan betänkandet och omröstningen.

4. Du rekommenderar ALDRIG ett parti och rangordnar dem aldrig. Ber
   användaren om en rekommendation förklarar du kort att du redovisar vad
   partierna föreslagit, och erbjuder en sakfrågejämförelse i stället.

5. Saknar ett parti motioner om ämnet i just detta material, skriv det
   som ett konstaterande ("Materialet innehåller ingen motion från SD om
   detta") — inte som att partiet saknar en åsikt.

FORM: svar på svenska, löpande text, två till fem stycken. Jämför du flera
partier, ett stycke per parti i samma ordning varje gång."""

# Steg 2: vad den vet. Q&A-par som lär modellen materialets semantik — utan
# dem riskerar den att övertolka ett yrkande som beslutad politik.
KUNSKAP = [
    ("Vad är en motion?",
     "Ett formellt förslag som en eller flera riksdagsledamöter lämnar in "
     "i riksdagen, i sitt partis namn. Den innehåller ett eller flera "
     "yrkanden — konkreta beslutsförslag — och en motivering till varför. "
     "En motion är partiets egen, unfiltrerade begäran, inte ett resultat "
     "av förhandling med andra partier (till skillnad från ett "
     "betänkande)."),
    ("Betyder 'Riksdagen ställer sig bakom det som anförs i motionen om X "
     "och tillkännager detta för regeringen' att X blev verklighet?",
     "Nej. Det är den formella texten för ett tillkännagivande — en "
     "signal till regeringen om vad riksdagen vill, inte en lag och inte "
     "ett bindande beslut. Även om riksdagen röstar ja till just den här "
     "formuleringen är regeringen inte juridiskt tvingad att agera på "
     "den. Motionen i sig säger heller ingenting om huruvida någon "
     "omröstning ens hållits."),
    ("Vad är skillnaden mellan ett yrkande och en sektion/motivering?",
     "Yrkandet är den korta, formella beslutsmeningen. Motiveringen (eller "
     "en namngiven underrubrik) är resonemanget bakom den — varför "
     "partiet vill det. Har ett yrkande ingen kopplad motivering i "
     "materialet betyder det bara att kopplingen inte gick att fastställa "
     "automatiskt, inte att partiet saknar ett skäl."),
    ("Ett parti har lämnat in nästan identiska yrkanden flera år i rad — "
     "vad betyder det?",
     "Bara att partiet återkommit till samma krav. Det kan bero på att "
     "det avslagits tidigare, att det inte kommit upp till behandling, "
     "eller andra skäl materialet inte visar. Dra ingen slutsats om "
     "UTFALLET — bara om att frågan är återkommande för partiet."),
]


def _källnummer(träffar):
    """Ett löpnummer per UNIKT källdokument (motion), i den ordning de
    först dyker upp i träfflistan. Utan detta numrerades citat efter
    position i träfflistan (1-8) — chunkar 3 och 4 kunde råka vara samma
    motion som chunk 1, försvinna ur källistan som dubbletter, men LLM:en
    hade redan citerat dem som [3]/[4] i kontexten den fick. Resultat: ett
    svar med källhänvisningar som saknade motsvarande rad i "Källor".
    Chunkar från samma motion delar nu nummer, så varje citerat nummer
    alltid går att slå upp."""
    nummer, näst = {}, 1
    for _, m in träffar:
        källa = f"motion {m['beteckning']} {m['rm']}"
        if källa not in nummer:
            nummer[källa] = näst
            näst += 1
    return nummer


def bygg_kontext(träffar, tak=MAX_KONTEXT_TECKEN):
    nummer = _källnummer(träffar)
    delar, n = ["HÄMTADE MOTIONSUTDRAG"], 0
    for _poäng, m in träffar:
        källa = f"motion {m['beteckning']} {m['rm']}"
        bit = (f"\n[{nummer[källa]}] {m['parti']} — {källa}"
               f"{' — ' + m['heading'] if m.get('heading') else ''}\n"
               f"{m['text']}")
        if m.get("sektion_text"):
            bit += f"\nMotivering: {m['sektion_text']}"
        bit += "\n"
        if n + len(bit) > tak:
            break
        delar.append(bit)
        n += len(bit)
    return "\n".join(delar)


def kallista(träffar):
    nummer = _källnummer(träffar)
    ut, sedda = [], set()
    for _p, m in träffar:
        källa = f"motion {m['beteckning']} {m['rm']}"
        if källa in sedda:
            continue
        sedda.add(källa)
        ut.append(f"[{nummer[källa]}] {m['parti']} — {källa}")
    return ut


def bygg_innehall(fråga, kontext):
    """Gemini-format: växlande user/model-turer, kontexten sist."""
    innehåll = []
    for f, s in KUNSKAP:
        innehåll.append({"role": "user", "parts": [{"text": f}]})
        innehåll.append({"role": "model", "parts": [{"text": s}]})
    innehåll.append({"role": "user", "parts": [{"text":
        f"KONTEXT\n{kontext}\n\nSLUT PÅ KONTEXT\n\nFRÅGA: {fråga}"}]})
    return innehåll


def fraga_gemini(innehåll, modell=GEMINI_MODELL, forsok=8):
    """Samma anropslogik (försök-om-vid-övergående-fel) som answer.py."""
    import random
    import time
    from google import genai
    from google.genai import types

    import os
    nyckel = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not nyckel:
        sys.exit("sätt GEMINI_API_KEY (nyckel från https://aistudio.google.com)")
    klient = genai.Client(api_key=nyckel)

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM, temperature=0.2,
        max_output_tokens=MAX_UT_TOKENS)

    OVERGAENDE = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE",
                  "500", "INTERNAL", "504", "DEADLINE_EXCEEDED")

    for n in range(forsok):
        try:
            svar = klient.models.generate_content(
                model=modell, contents=innehåll, config=config)
        except Exception as fel:
            text = str(fel)
            if not any(kod in text for kod in OVERGAENDE):
                raise
            if n == forsok - 1:
                break
            paus = min(60, 2 ** n * 4) * (0.5 + random.random())
            print(f"  (övergående fel, försöker igen om {paus:.0f}s …)",
                  file=sys.stderr)
            time.sleep(paus)
            continue
        if not svar.text:
            skal = ""
            if getattr(svar, "candidates", None):
                skal = str(getattr(svar.candidates[0], "finish_reason", "") or "")
            return f"[inget svar genererades — finish_reason: {skal or 'okänt'}]"
        return svar.text

    sys.exit(f"gav upp efter {forsok} försök mot {modell}")


def svara(idx, fråga, k=8, torrkor=False):
    partier = parti_i_fragan(fråga)
    träffar = idx.search(fråga, k=k, parti=partier or None)
    kontext = bygg_kontext(träffar)
    innehåll = bygg_innehall(fråga, kontext)
    if torrkor:
        return None, träffar, innehåll
    return fraga_gemini(innehåll), träffar, innehåll


def skriv(svar, träffar, innehåll, torrkor):
    if torrkor:
        print(innehåll[-1]["parts"][0]["text"])
        return
    print(svar)
    print("\nKällor:")
    for rad in kallista(träffar):
        print(" ", rad)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def inspect(path):
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


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)

    cmd = sys.argv[1]
    if cmd == "inspect":
        inspect(Path(sys.argv[2]))
    elif cmd == "build":
        if len(sys.argv) != 4:
            sys.exit(__doc__)
        build(sys.argv[2], sys.argv[3])
    elif cmd == "embed":
        if len(sys.argv) not in (4, 5):
            sys.exit(__doc__)
        stub = len(sys.argv) == 5 and sys.argv[4] == "--stub"
        embed(sys.argv[2], sys.argv[3], stub=stub)
    elif cmd == "search":
        if len(sys.argv) != 4:
            sys.exit(__doc__)
        idx = Index(sys.argv[2])
        fråga = sys.argv[3]
        partier = parti_i_fragan(fråga)
        if partier:
            print(f"[partifilter: {', '.join(partier)}]")
        visa(idx.search(fråga, parti=partier or None))
    elif cmd == "shell":
        if len(sys.argv) != 3:
            sys.exit(__doc__)
        idx = Index(sys.argv[2])
        print(f"{idx.info['n']} chunkar, modell {idx.info['model']}. Tom rad avslutar.")
        while True:
            try:
                fråga = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not fråga:
                break
            partier = parti_i_fragan(fråga)
            if partier:
                print(f"[partifilter: {', '.join(partier)}]")
            visa(idx.search(fråga, parti=partier or None))
    elif cmd == "fraga":
        if len(sys.argv) not in (4, 5):
            sys.exit(__doc__)
        torrkor = len(sys.argv) == 5 and sys.argv[4] == "--torrkor"
        idx = Index(sys.argv[2])
        skriv(*svara(idx, sys.argv[3], torrkor=torrkor), torrkor=torrkor)
    elif cmd == "chatt":
        if len(sys.argv) not in (3, 4):
            sys.exit(__doc__)
        torrkor = len(sys.argv) == 4 and sys.argv[3] == "--torrkor"
        idx = Index(sys.argv[2])
        print(f"{idx.info['n']} chunkar. Tom rad avslutar.")
        while True:
            try:
                fråga = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not fråga:
                break
            skriv(*svara(idx, fråga, torrkor=torrkor), torrkor=torrkor)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()