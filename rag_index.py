"""
rag_index.py — bygger och söker i det gemensamma indexet över båda lagren.

SAID  = vad partierna säger på sina hemsidor   (out/positions_said_*.jsonl)
DID   = vad som står i riksdagens betänkanden  (out/chunks.jsonl)

Varje passage får en kontextrad före själva texten. Kontextraden är en
wordalisation av metadatan — "Reservation 1 (C) i AU10, 2022/23, punkt 1" —
så att både embeddingen och LLM:en ser vad texten ÄR, inte bara vad den säger.

Sökningen är hybrid: BM25 (exakta ord, beteckningar, siffror) och vektorer
(betydelse) slås ihop med viktad Reciprocal Rank Fusion. Träffarna kvoteras
per parti, annars äter ett ordrikt parti hela kontextfönstret och
"jämförelsen" blir en sammanfattning av det parti som skriver mest.

Kör:
    python rag_index.py build                      # ~20 min med e5-small
    python rag_index.py build --stub               # rökttest utan modell
    python rag_index.py build --bara-bm25          # bygg om BM25, behåll vektorer
    python rag_index.py search "vinster i välfärden"
    python rag_index.py search "klimat" --parti V,SD --per-parti 3
    python rag_index.py search "AU10 2022/23" --metod bm25 --layer did
    python rag_index.py shell
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import re
import sys
from collections import Counter, defaultdict

SAID_GLOB = "out/positions_said_*.jsonl"
DID_CHUNKS = "out/chunks.jsonl"
INDEX_DIR = "out/index"

MODEL_NAME = "intfloat/multilingual-e5-large"
BATCH = 32

# Passagelängd. e5 klipper vid 512 tokens ≈ 1 600 tecken svenska; vi håller
# oss under det med marginal och låter styckena överlappa så att en mening
# inte kan hamna mellan två passager.
MAX_CHARS = 1200
OVERLAP = 150
RRF_K = 60
# RRF väger normalt de två listorna lika. Vår utvärdering visar att BM25 är
# klart svagare än vektorlistan på naturligt formulerade frågor, och att en
# likaviktad sammanslagning då DRAR NER resultatet under ren vektorsökning.
# BM25 får därför lägre vikt: den ska rädda identifierarsökningar
# ("AU10 2022/23", "reservation 14"), inte styra ämnessökningen.
RRF_VIKT_BM25 = 0.35
RRF_VIKT_VEKTOR = 1.0

PARTY_NAMES = {
    "S": "Socialdemokraterna", "M": "Moderaterna", "SD": "Sverigedemokraterna",
    "C": "Centerpartiet", "V": "Vänsterpartiet", "KD": "Kristdemokraterna",
    "MP": "Miljöpartiet", "L": "Liberalerna",
}
SECTION_SV = {
    "summary": "Sammanfattning", "decision": "Utskottets förslag till beslut",
    "deliberation": "Utskottets överväganden", "reservation": "Reservation",
    "dissent": "Särskilt yttrande",
}

# Ord som bara skapar brus i BM25. Medvetet kort — vi vill inte råka
# filtrera bort politiskt laddade ord.
STOPP = set("""och att det som en för av på är den till med de i om så har inte
den där vi men kan ska har hade blir blev vara var vid ett den denna detta dessa
under efter mot från utan även samt då när där vilket vilka man sig sin sitt
dess deras våra vår vårt eller än mer mest också bara alla andra""".split())

WORD_RE = re.compile(r"[a-zåäöéA-ZÅÄÖÉ0-9]+")

# Partinamn är brus i söksträngen så snart partiet är ett metadatafilter:
# ordet "miljöpartiet" står på nästan varje mp.se-sida och dränker ämnesordet.
PARTINAMN = set()
for _k, _v in PARTY_NAMES.items():
    PARTINAMN |= {_k.lower(), _v.lower()}
PARTINAMN |= {"moderata", "samlingspartiet", "socialdemokrater",
              "sverigedemokrater", "kristdemokrater", "centern", "partiet"}

# Lätt svensk stamning. Ordnad längst först; vi kapar bara om stammen blir
# minst fyra tecken, så "det" och "stat" lämnas i fred.
SUFFIX = ["ernas", "arnas", "ornas", "andet", "arna", "erna", "orna",
          "ande", "ende", "aste", "ades", "ade", "are", "ast", "ens",
          "ets", "er", "ar", "or", "en", "et", "na", "as", "es"]
MIN_STAM = 4


def stam(ord_):
    for suf in SUFFIX:
        if ord_.endswith(suf) and len(ord_) - len(suf) >= MIN_STAM:
            return ord_[: -len(suf)]
    return ord_


# --------------------------------------------------------------------------
# 1. läs in och normalisera båda lagren
# --------------------------------------------------------------------------

def split_text(text, max_chars=MAX_CHARS, overlap=OVERLAP):
    """Dela lång text på stycke- och meningsgränser, aldrig mitt i ett ord."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    # Först på styckegränser, sedan på meningsgränser om ett stycke är för långt.
    bitar = []
    for stycke in re.split(r"\n\s*\n", text):
        stycke = stycke.strip()
        if not stycke:
            continue
        if len(stycke) <= max_chars:
            bitar.append(stycke)
            continue
        meningar = re.split(r"(?<=[.!?])\s+", stycke)
        buf = ""
        for m in meningar:
            while len(m) > max_chars:          # en enda jättemening
                bitar.append(m[:max_chars])
                m = m[max_chars - overlap:]
            if len(buf) + len(m) + 1 <= max_chars:
                buf = f"{buf} {m}".strip()
            else:
                if buf:
                    bitar.append(buf)
                buf = m
        if buf:
            bitar.append(buf)

    # Slå ihop småbitar och lägg på överlapp.
    ut = []
    buf = ""
    for b in bitar:
        if len(buf) + len(b) + 2 <= max_chars:
            buf = f"{buf}\n\n{b}".strip()
        else:
            if buf:
                ut.append(buf)
            buf = b
    if buf:
        ut.append(buf)

    if overlap and len(ut) > 1:
        med_overlap = [ut[0]]
        for i in range(1, len(ut)):
            svans = ut[i - 1][-overlap:]
            mellanslag = svans.find(" ")          # kapa fram till ordgräns
            svans = svans[mellanslag + 1:] if mellanslag != -1 else ""
            med_overlap.append(f"{svans} {ut[i]}".strip())
        ut = med_overlap
    return ut


def said_context(d):
    parti = d.get("parti", "")
    namn = PARTY_NAMES.get(parti, parti)
    rubrik = d.get("heading") or d.get("sakfraga") or ""
    return f"{namn} ({parti}) om {d.get('sakfraga', '')}: {rubrik}".strip()


def did_context(d):
    sek = SECTION_SV.get(d.get("section", ""), d.get("section", ""))
    bitar = [sek]
    if d.get("section") in ("reservation", "dissent") and d.get("number"):
        bitar[0] = f"{sek} {d['number']}"
    partier = [p for p in str(d.get("parties") or "").split(";") if p]
    if partier:
        bitar[0] += f" ({', '.join(partier)})"
    var = f"{d.get('beteckning', '')} {d.get('rm', '')}"
    if d.get("punkter"):
        var += f" punkt {d['punkter']}"
    bitar.append(var.strip())
    if d.get("doc_title"):
        bitar.append(d["doc_title"])
    if d.get("heading"):
        bitar.append(d["heading"])
    return " — ".join(b for b in bitar if b)


def load_records(motions_path=None):
    """Båda lagren -> en lista av passager med gemensamt schema."""
    poster = []

    for path in sorted(glob.glob(SAID_GLOB)):
        for line in open(path, encoding="utf-8"):
            d = json.loads(line)
            ctx = said_context(d)
            for i, bit in enumerate(split_text(d.get("text", ""))):
                poster.append({
                    "id": f"{d['chunk_id']}#{i}",
                    "parent": d["chunk_id"],
                    "layer": "said",
                    "parti": d.get("parti", ""),
                    "sakfraga": d.get("sakfraga", ""),
                    "url": d.get("url", ""),
                    "lastmod": d.get("lastmod"),
                    "kontext": ctx,
                    "text": bit,
                })

    for line in open(DID_CHUNKS, encoding="utf-8"):
        d = json.loads(line)
        ctx = did_context(d)
        partier = [p for p in str(d.get("parties") or "").split(";") if p]
        for i, bit in enumerate(split_text(d.get("text", ""))):
            poster.append({
                "id": f"{d['chunk_id']}#{i}",
                "parent": d["chunk_id"],
                "layer": "did",
                "parti": ";".join(partier),
                "section": d.get("section", ""),
                "rm": d.get("rm", ""),
                "beteckning": d.get("beteckning", ""),
                "punkt": str(d.get("punkter") or ""),
                "utskott": d.get("utskott", ""),
                "number": d.get("number"),
                "doc_title": d.get("doc_title", ""),
                "doc_verdict": d.get("doc_verdict", ""),
                "kontext": ctx,
                "text": bit,
            })
    if motions_path:
        with open(motions_path, encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                authors = ", ".join(f"{p['namn']} ({p['partibet']})" for p in d['undertecknare'])
                ctx = (f"Motion {d['rm']}:{d['beteckning']} — {d['heading']} — "
                       f"{d.get('motionstyp', '')}, inlämnad {d.get('datum', 'okänt')} — "
                       f"Förslag av {authors}. Inte ett beslut eller belägg för hela partiets stöd.")
                for i, bit in enumerate(split_text(d['text'])):
                    poster.append({**d, "id": f"{d['chunk_id']}#{i}",
                                   "parent": d['chunk_id'], "kontext": ctx, "text": bit})
    return poster


def passage_text(p):
    """Det som faktiskt embeddas: kontextraden och sedan texten."""
    if p['layer'] == 'motion':
        # Långa namnlistor får inte tränga undan själva förslaget i sökmodellen.
        # Full avsändarinformation finns kvar i kontexten till Gemini.
        context = (f"Motion {p['rm']}:{p['beteckning']} — {p['heading']} — "
                   f"förslag från ledamöter ({p['parti']})")
        return f"passage: {context}\n{p['text']}"
    return f"passage: {p['kontext']}\n{p['text']}"


# --------------------------------------------------------------------------
# 2. BM25
# --------------------------------------------------------------------------

def tokenize(s, ta_bort_partinamn=False):
    ord_ = (t.lower() for t in WORD_RE.findall(s))
    ord_ = (w for w in ord_ if w not in STOPP)
    if ta_bort_partinamn:
        ord_ = (w for w in ord_ if w not in PARTINAMN)
    return [stam(w) for w in ord_]


class BM25:
    """Standard BM25 med inverterat index. Ren stdlib — inget beroende."""

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

    # Vi picklar data, inte objektet. Ett picklat objekt går bara att läsa
    # tillbaka från samma modul det skapades i — annars kraschar importen
    # med "Can't get attribute 'BM25' on <module '__main__'>".
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

    def search(self, query, limit=200, ta_bort_partinamn=False):
        poäng = defaultdict(float)
        for term in tokenize(query, ta_bort_partinamn):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.inv.get(term, ()):
                norm = 1 - self.b + self.b * self.längder[i] / self.medel
                poäng[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return sorted(poäng.items(), key=lambda kv: -kv[1])[:limit]


# --------------------------------------------------------------------------
# 3. embeddings
# --------------------------------------------------------------------------

def stub_vectors(texts, dim=256):
    """Deterministisk fejk-encoder för rökttest. INTE semantisk."""
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
    import torch
    enhet = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"laddar {model_name} på {enhet} …")
    modell = SentenceTransformer(model_name, device=enhet)
    modell.max_seq_length = 384    # ~1 200 tecken svenska ryms; default 512
    return modell.encode(
        texts, batch_size=BATCH, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    ).astype("float32")


def encode_query(text, stub=False, model_name=MODEL_NAME):
    q = f"query: {text}"
    if stub:
        return stub_vectors([q])[0]
    from sentence_transformers import SentenceTransformer
    import torch
    enhet = "mps" if torch.backends.mps.is_available() else "cpu"
    modell = SentenceTransformer(model_name, device=enhet)
    return modell.encode([q], normalize_embeddings=True,
                         convert_to_numpy=True).astype("float32")[0]


# --------------------------------------------------------------------------
# 4. build
# --------------------------------------------------------------------------

def build(stub=False, model_name=MODEL_NAME, out_dir=INDEX_DIR,
          bara_bm25=False, motions_path=None, base_index=None):
    import numpy as np

    poster = load_records(motions_path)
    base = None
    if base_index:
        if os.path.realpath(base_index) == os.path.realpath(out_dir):
            raise ValueError("Basindex och nytt index måste vara olika mappar")
        base = Index(base_index)
        if stub or base.stub or base.meta != poster[:len(base.meta)]:
            raise ValueError("Basindexets texter/ordning matchar inte. Bygg ett nytt index utan --base-index.")
        model_name = base.info['model']
    if bara_bm25:
        meta_path = os.path.join(out_dir, 'meta.jsonl')
        if not os.path.exists(meta_path):
            raise ValueError("--bara-bm25 kräver ett befintligt index")
        previous = [json.loads(line) for line in open(meta_path, encoding='utf-8')]
        if previous != poster:
            raise ValueError("Texterna har ändrats: även vektorerna måste byggas om")
    print(f"{len(poster)} passager")
    lager = Counter(p["layer"] for p in poster)
    print("  " + "  ".join(f"{k} {v}" for k, v in sorted(lager.items())))

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "meta.jsonl"), "w", encoding="utf-8") as fh:
        for p in poster:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    print("bygger BM25 …")
    docs = [tokenize(f"{p['kontext']} {p['text']}") for p in poster]
    BM25(docs).save(os.path.join(out_dir, "bm25.pkl"))

    if bara_bm25:
        print("hoppar över embeddingen (--bara-bm25); vectors.npy rörs inte")
        return poster

    print("embeddar …")
    new = poster[len(base.meta):] if base else poster
    if not new and base:
        vek = base.vek
    else:
        vek = encode([passage_text(p) for p in new], stub=stub,
                     model_name=model_name)
        if base:
            vek = np.concatenate((base.vek, vek))
    np.save(os.path.join(out_dir, "vectors.npy"), vek)

    with open(os.path.join(out_dir, "info.json"), "w", encoding="utf-8") as fh:
        json.dump({"model": "stub" if stub else model_name,
                   "dim": int(vek.shape[1]), "n": len(poster),
                   "max_chars": MAX_CHARS, "overlap": OVERLAP,
                   "layers": sorted(lager)}, fh,
                  ensure_ascii=False, indent=2)

    print(f"\nskrev {out_dir}/  ({vek.shape[0]} × {vek.shape[1]})")
    return poster


# --------------------------------------------------------------------------
# 5. search
# --------------------------------------------------------------------------

class Index:
    """Håller index i minnet. Använd 'shell' för flera frågor i rad —
    annars laddas embeddingmodellen om vid varje CLI-anrop."""

    def __init__(self, out_dir=INDEX_DIR):
        import numpy as np
        if not os.path.exists(os.path.join(out_dir, "info.json")):
            sys.exit(f"hittar inget index i {out_dir} — kör 'build' först")
        self.info = json.load(open(os.path.join(out_dir, "info.json")))
        self.meta = [json.loads(l) for l in
                     open(os.path.join(out_dir, "meta.jsonl"), encoding="utf-8")]
        self.vek = np.load(os.path.join(out_dir, "vectors.npy"))
        self.bm = BM25.load(os.path.join(out_dir, "bm25.pkl"))
        self.stub = self.info["model"] == "stub"

    def _encode_query(self, fråga):
        """Modellen laddas en gång per Index-instans, inte per fråga."""
        if self.stub:
            return stub_vectors([f"query: {fråga}"])[0]
        if not hasattr(self, "_modell"):
            from sentence_transformers import SentenceTransformer
            import torch
            enhet = "mps" if torch.backends.mps.is_available() else "cpu"
            self._modell = SentenceTransformer(self.info["model"], device=enhet)
        return self._modell.encode([f"query: {fråga}"], normalize_embeddings=True,
                                   convert_to_numpy=True).astype("float32")[0]

    def search(self, fråga, k=8, parti=None, layer=None, per_parti=None,
               kandidater=300, metod="hybrid", vikt_bm25=None, rm=None):
        """metod: 'hybrid' (BM25 + vektor), 'bm25' eller 'vektor'.

        De två rena lägena finns för ablationen i rapporten — och 'bm25'
        kräver ingen modell, vilket gör det möjligt att testa allt annat
        utan att ladda 2 GB.
        """
        import numpy as np

        w_bm = RRF_VIKT_BM25 if vikt_bm25 is None else vikt_bm25
        allowed = {i for i, m in enumerate(self.meta)
                   if (not layer or m['layer'] == layer)
                   and (not rm or m.get('rm') == rm)
                   and (not parti or set(m.get('parti', '').split(';')) & set(parti))}
        if not allowed:
            return []
        listor = []
        if metod in ("hybrid", "bm25"):
            listor.append((1.0 if metod == "bm25" else w_bm,
                           [(i, score) for i, score in self.bm.search(
                               fråga, limit=len(self.meta), ta_bort_partinamn=bool(parti))
                            if i in allowed][:kandidater]))
        if metod in ("hybrid", "vektor"):
            qv = self._encode_query(fråga)
            sim = self.vek @ qv
            listor.append((RRF_VIKT_VEKTOR,
                           [(int(i), float(sim[i]))
                            for i in np.argsort(-sim) if int(i) in allowed][:kandidater]))

        # Reciprocal Rank Fusion: rangordning slås ihop utan att vi behöver
        # normalisera två helt olika poängskalor mot varandra.
        rrf = defaultdict(float)
        for vikt, lista in listor:
            for rang, (i, _) in enumerate(lista):
                rrf[i] += vikt / (RRF_K + rang + 1)

        träffar = []
        for i, poäng in sorted(rrf.items(), key=lambda kv: -kv[1]):
            m = self.meta[i]
            if layer and m["layer"] != layer:
                continue
            if parti:
                egna = set(p for p in m.get("parti", "").split(";") if p)
                if not egna & set(parti):
                    continue
            träffar.append((poäng, m))

        if per_parti:
            per = defaultdict(int)
            kvoterat = []
            for poäng, m in träffar:
                nycklar = [p for p in m.get("parti", "").split(";") if p] or ["—"]
                if all(per[n] >= per_parti for n in nycklar):
                    continue
                for n in nycklar:
                    per[n] += 1
                kvoterat.append((poäng, m))
            träffar = kvoterat

        return träffar[:k]


def visa(träffar):
    if not träffar:
        print("inga träffar")
        return
    for n, (poäng, m) in enumerate(träffar, 1):
        källa = m.get("url") or f"{m.get('beteckning', '')} {m.get('rm', '')}"
        text = m["text"].replace("\n", " ")
        print(f"\n{n:2}. [{poäng:.4f}] {m['layer'].upper()}  {m['kontext']}")
        print(f"    {text[:260]}{'…' if len(text) > 260 else ''}")
        print(f"    {källa}   id={m['id']}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--stub", action="store_true",
                   help="rökttest med fejk-encoder, ingen modell laddas")
    b.add_argument("--model", default=MODEL_NAME)
    b.add_argument("--out", default=INDEX_DIR)
    b.add_argument("--motions", help="Bearbetade motioner, t.ex. out/motions.jsonl")
    b.add_argument("--base-index", help="Återanvänd oförändrade basvektorer i ett separat nytt index")
    b.add_argument("--bara-bm25", action="store_true",
                   help="bygg om meta + BM25 utan att röra vectors.npy")

    s = sub.add_parser("search")
    s.add_argument("fraga")
    s.add_argument("-k", type=int, default=8)
    s.add_argument("--parti", default="", help="t.ex. V,SD")
    s.add_argument("--layer", choices=["said", "did", "motion"])
    s.add_argument("--rm", help="Avgränsa till ett riksmöte, t.ex. 2023/24")
    s.add_argument("--per-parti", type=int, default=None,
                   help="max antal träffar per parti")
    s.add_argument("--index", default=INDEX_DIR)
    s.add_argument("--metod", choices=["hybrid", "bm25", "vektor"], default="hybrid")
    s.add_argument("--vikt-bm25", type=float, default=None,
                   help=f"BM25:s vikt i RRF (standard {RRF_VIKT_BM25})")

    sh = sub.add_parser("shell", help="flera frågor i rad, modellen laddas en gång")
    sh.add_argument("-k", type=int, default=8)
    sh.add_argument("--index", default=INDEX_DIR)

    a = ap.parse_args()
    if a.cmd == "build":
        build(stub=a.stub, model_name=a.model, out_dir=a.out,
              bara_bm25=a.bara_bm25, motions_path=a.motions, base_index=a.base_index)
    elif a.cmd == "shell":
        idx = Index(a.index)
        print(f"{idx.info['n']} passager, modell {idx.info['model']}. "
              f"Tom rad avslutar.")
        while True:
            try:
                fråga = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not fråga:
                break
            visa(idx.search(fråga, k=a.k))
    else:
        idx = Index(a.index)
        partier = [p.strip().upper() for p in a.parti.split(",") if p.strip()]
        visa(idx.search(a.fraga, k=a.k, parti=partier or None,
                        layer=a.layer, per_parti=a.per_parti, metod=a.metod,
                        vikt_bm25=a.vikt_bm25, rm=a.rm))


if __name__ == "__main__":
    main()
