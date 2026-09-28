"""
rag_index.py — builds and searches the shared index over all three layers.

SAID    = what the parties say on their websites  (out/positions_said_*.jsonl)
DID     = what's in the Riksdag's committee reports (out/chunks.jsonl)
MOTION  = the parties' formal proposals            (out/motion_chunks.jsonl)

Every passage gets a context line in front of its text. The context line is a
wordalisation of the metadata — "Reservation 1 (C) — AU10 2022/23 punkt 1" —
so both the embedding and the LLM see what the text IS, not just what it says.
The context lines are Swedish on purpose: they are embedded together with the
Swedish text, and changing them changes every vector.

Search is hybrid: BM25 (exact words, beteckningar, numbers) and vectors
(meaning) are merged with weighted Reciprocal Rank Fusion.

Run:
    python rag_index.py build                      # embeds only new/changed passages
    python rag_index.py build --stub               # smoke test without a model
    python rag_index.py build --bm25-only          # rebuild BM25, keep vectors
    python rag_index.py search "vinster i välfärden"
    python rag_index.py search "klimat" --party V,SD --per-party 3
    python rag_index.py search "AU10 2022/23" --method bm25 --layer did
    python rag_index.py shell
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import pickle
import re
import sys
from collections import Counter, defaultdict

SAID_GLOB = "out/positions_said_*.jsonl"
DID_CHUNKS = "out/chunks.jsonl"
MOTION_CHUNKS = "out/motion_chunks.jsonl"
INDEX_DIR = "out/index"

MODEL_NAME = "intfloat/multilingual-e5-large"
BATCH = 32

# Passage length. e5 truncates at 512 tokens ≈ 1,600 characters of Swedish; we
# stay well under that and let pieces overlap so a sentence can't fall between
# two passages.
MAX_CHARS = 1200
OVERLAP = 150
RRF_K = 60
# RRF normally weights the two lists equally. Our evaluation shows BM25 is
# clearly weaker than the vector list on naturally phrased questions, and an
# equally weighted merge then PULLS the result below pure vector search.
# BM25 therefore gets a lower weight: it's there to rescue identifier lookups
# ("AU10 2022/23", "reservation 14"), not to drive topical ranking.
RRF_WEIGHT_BM25 = 0.35
RRF_WEIGHT_VECTOR = 1.0

PARTY_NAMES = {
    "S": "Socialdemokraterna", "M": "Moderaterna", "SD": "Sverigedemokraterna",
    "C": "Centerpartiet", "V": "Vänsterpartiet", "KD": "Kristdemokraterna",
    "MP": "Miljöpartiet", "L": "Liberalerna",
}
# Swedish section labels for the DID context line (embedded text — keep as is).
SECTION_SV = {
    "summary": "Sammanfattning", "decision": "Utskottets förslag till beslut",
    "deliberation": "Utskottets överväganden", "reservation": "Reservation",
    "dissent": "Särskilt yttrande",
}

# Swedish words that only add noise in BM25. Deliberately short — we don't
# want to accidentally filter out politically loaded words.
STOPWORDS = set("""och att det som en för av på är den till med de i om så har inte
den där vi men kan ska har hade blir blev vara var vid ett den denna detta dessa
under efter mot från utan även samt då när där vilket vilka man sig sin sitt
dess deras våra vår vårt eller än mer mest också bara alla andra""".split())

WORD_RE = re.compile(r"[a-zåäöéA-ZÅÄÖÉ0-9]+")

# Party names are noise in the query as soon as the party is a metadata
# filter: the word "miljöpartiet" is on almost every mp.se page and drowns
# out the topic word.
PARTY_NAME_WORDS = set()
for _code, _name in PARTY_NAMES.items():
    PARTY_NAME_WORDS |= {_code.lower(), _name.lower()}
PARTY_NAME_WORDS |= {"moderata", "samlingspartiet", "socialdemokrater",
                     "sverigedemokrater", "kristdemokrater", "centern", "partiet"}

# Light Swedish stemming. Ordered longest first; we only cut if the stem
# keeps at least four characters, so "det" and "stat" are left alone.
SUFFIXES = ["ernas", "arnas", "ornas", "andet", "arna", "erna", "orna",
            "ande", "ende", "aste", "ades", "ade", "are", "ast", "ens",
            "ets", "er", "ar", "or", "en", "et", "na", "as", "es"]
MIN_STEM = 4


def stem(word):
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= MIN_STEM:
            return word[: -len(suffix)]
    return word


YRKANDE_WRAPPER_PATTERNS = [
    re.compile(
        r"^Riksdagen ställer sig bakom det som anförs i motionen om (att )?"
        r"(?P<core>.+?)"
        r"[,.]?\s*och\s+(detta\s+tillkännager\s+riksdagen|tillkännager\s+detta)\s+för\s+regeringen\.?\s*$",
        re.IGNORECASE),
    re.compile(
        r"^Riksdagen avslår regeringens förslag\s+"
        r"(avseende|i\s+den\s+del\s+som\s+avser)\s+"
        r"(?P<core>.+?)\.?\s*$",
        re.IGNORECASE),
]


def yrkande_core(text):
    """Strips the boilerplate ('Riksdagen ställer sig bakom...') from a
    yrkande, to leave the embedding more room for the actual content."""
    for pattern in YRKANDE_WRAPPER_PATTERNS:
        m = pattern.match(text)
        if m:
            return m.group("core").strip()
    return text


# --------------------------------------------------------------------------
# 1. load and normalise all layers
# --------------------------------------------------------------------------

def split_text(text, max_chars=MAX_CHARS, overlap=OVERLAP):
    """Split long text at paragraph and sentence boundaries, never mid-word."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    # Paragraph boundaries first, then sentence boundaries if a paragraph is too long.
    pieces = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            pieces.append(paragraph)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        buf = ""
        for s in sentences:
            while len(s) > max_chars:          # one giant sentence
                pieces.append(s[:max_chars])
                s = s[max_chars - overlap:]
            if len(buf) + len(s) + 1 <= max_chars:
                buf = f"{buf} {s}".strip()
            else:
                if buf:
                    pieces.append(buf)
                buf = s
        if buf:
            pieces.append(buf)

    # Merge small pieces and add the overlap.
    out = []
    buf = ""
    for p in pieces:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = f"{buf}\n\n{p}".strip()
        else:
            if buf:
                out.append(buf)
            buf = p
    if buf:
        out.append(buf)

    if overlap and len(out) > 1:
        with_overlap = [out[0]]
        for i in range(1, len(out)):
            tail = out[i - 1][-overlap:]
            space = tail.find(" ")          # cut forward to a word boundary
            tail = tail[space + 1:] if space != -1 else ""
            with_overlap.append(f"{tail} {out[i]}".strip())
        out = with_overlap
    return out


def said_context(d):
    party = d.get("parti", "")
    name = PARTY_NAMES.get(party, party)
    heading = d.get("heading") or d.get("sakfraga") or ""
    ctx = f"{name} ({party}) om {d.get('sakfraga', '')}: {heading}".strip()
    # The party's own A–Ö labels ("Artificiell intelligens, AI" on the page
    # "Digitalisering") — in the context line so every piece of a split page gets them.
    extra = [label for label in d.get("labels", []) if label.lower() != heading.lower()]
    return f"{ctx} ({'; '.join(extra)})" if extra else ctx


def did_context(d):
    section = SECTION_SV.get(d.get("section", ""), d.get("section", ""))
    parts = [section]
    if d.get("section") in ("reservation", "dissent") and d.get("number"):
        parts[0] = f"{section} {d['number']}"
    parties = [p for p in str(d.get("parties") or "").split(";") if p]
    if parties:
        parts[0] += f" ({', '.join(parties)})"
    where = f"{d.get('beteckning', '')} {d.get('rm', '')}"
    if d.get("punkter"):
        where += f" punkt {d['punkter']}"
    parts.append(where.strip())
    if d.get("doc_title"):
        parts.append(d["doc_title"])
    if d.get("heading"):
        parts.append(d["heading"])
    return " — ".join(p for p in parts if p)


def motion_context(d):
    party = d.get("parti", "")
    name = PARTY_NAMES.get(party, party)
    ctx = f"{name} ({party}), motion {d.get('beteckning', '')} {d.get('rm', '')}"
    if d.get("heading"):
        ctx += f", om {d['heading']}"
    return ctx.strip()


def load_records():
    """All layers -> one list of passages with a shared schema.

    The JSON field names (parti, kontext, sakfraga, …) are the index's data
    schema and stay Swedish: meta.jsonl and every reader of it depend on them.
    """
    records = []

    for path in sorted(glob.glob(SAID_GLOB)):
        for line in open(path, encoding="utf-8"):
            d = json.loads(line)
            ctx = said_context(d)
            for i, piece in enumerate(split_text(d.get("text", ""))):
                records.append({
                    "id": f"{d['chunk_id']}#{i}",
                    "parent": d["chunk_id"],
                    "layer": "said",
                    "parti": d.get("parti", ""),
                    "sakfraga": d.get("sakfraga", ""),
                    "url": d.get("url", ""),
                    "lastmod": d.get("lastmod"),
                    "kontext": ctx,
                    "text": piece,
                })

    for line in open(DID_CHUNKS, encoding="utf-8"):
        d = json.loads(line)
        ctx = did_context(d)
        parties = [p for p in str(d.get("parties") or "").split(";") if p]
        for i, piece in enumerate(split_text(d.get("text", ""))):
            records.append({
                "id": f"{d['chunk_id']}#{i}",
                "parent": d["chunk_id"],
                "layer": "did",
                "parti": ";".join(parties),
                "section": d.get("section", ""),
                "rm": d.get("rm", ""),
                "beteckning": d.get("beteckning", ""),
                "punkt": str(d.get("punkter") or ""),
                "utskott": d.get("utskott", ""),
                "number": d.get("number"),
                "doc_title": d.get("doc_title", ""),
                "doc_verdict": d.get("doc_verdict", ""),
                "kontext": ctx,
                "text": piece,
            })

    if os.path.exists(MOTION_CHUNKS):
        for line in open(MOTION_CHUNKS, encoding="utf-8"):
            d = json.loads(line)
            ctx = motion_context(d)
            for i, piece in enumerate(split_text(d.get("text", ""))):
                records.append({
                    "id": f"{d['chunk_id']}#{i}",
                    "parent": d["chunk_id"],
                    "layer": "motion",
                    "parti": d.get("parti", ""),
                    "rm": d.get("rm", ""),
                    "beteckning": d.get("beteckning", ""),
                    "doc_title": d.get("doc_title", ""),
                    "heading": d.get("heading", ""),
                    "sektion_text": d.get("sektion_text", ""),
                    "yrkande_nr": d.get("yrkande_nr", 0),
                    "undertecknare": d.get("undertecknare", ""),
                    "kontext": ctx,
                    "text": piece,
                    "extra_bm25": d.get("sektion_text", ""),
                })
    return records


def passage_text(p):
    """What actually gets embedded: the context line, then the text.
    For motions the boilerplate is stripped from the yrkande."""
    text = yrkande_core(p["text"]) if p.get("layer") == "motion" else p["text"]
    return f"passage: {p['kontext']}\n{text}"


# --------------------------------------------------------------------------
# 2. BM25
# --------------------------------------------------------------------------

def tokenize(s, drop_party_names=False):
    words = (t.lower() for t in WORD_RE.findall(s))
    words = (w for w in words if w not in STOPWORDS)
    if drop_party_names:
        words = (w for w in words if w not in PARTY_NAME_WORDS)
    return [stem(w) for w in words]


class BM25:
    """Standard BM25 with an inverted index. Pure stdlib — no dependency."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.lengths = [len(d) for d in docs]
        self.avg_len = sum(self.lengths) / self.N if self.N else 0.0
        self.inv = defaultdict(list)          # term -> [(doc_idx, tf), ...]
        df = Counter()
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                self.inv[term].append((i, tf))
                df[term] += 1
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()
        }

    # We pickle data, not the object. A pickled object can only be read back
    # from the same module it was created in — otherwise the import crashes
    # with "Can't get attribute 'BM25' on <module '__main__'>".
    # The keys "längder"/"medel" are the on-disk format of bm25.pkl; renaming
    # them would break every existing index file.
    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump({"k1": self.k1, "b": self.b, "N": self.N,
                         "längder": self.lengths, "medel": self.avg_len,
                         "inv": dict(self.inv), "idf": self.idf},
                        fh, protocol=4)

    @classmethod
    def load(cls, path):
        d = pickle.load(open(path, "rb"))
        o = cls.__new__(cls)
        o.k1, o.b, o.N = d["k1"], d["b"], d["N"]
        o.lengths, o.avg_len = d["längder"], d["medel"]
        o.inv, o.idf = d["inv"], d["idf"]
        return o

    def search(self, query, limit=200, drop_party_names=False):
        scores = defaultdict(float)
        for term in tokenize(query, drop_party_names):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.inv.get(term, ()):
                norm = 1 - self.b + self.b * self.lengths[i] / self.avg_len
                scores[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return sorted(scores.items(), key=lambda kv: -kv[1])[:limit]


# --------------------------------------------------------------------------
# 3. embeddings
# --------------------------------------------------------------------------

def stub_vectors(texts, dim=256):
    """Deterministic fake encoder for smoke tests. NOT semantic."""
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
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {model_name} on {device} …")
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = 384    # ~1,200 characters of Swedish fit; default 512
    return model.encode(
        texts, batch_size=BATCH, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    ).astype("float32")


def encode_query(text, stub=False, model_name=MODEL_NAME):
    q = f"query: {text}"
    if stub:
        return stub_vectors([q])[0]
    from sentence_transformers import SentenceTransformer
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    return model.encode([q], normalize_embeddings=True,
                        convert_to_numpy=True).astype("float32")[0]


# --------------------------------------------------------------------------
# 4. build
# --------------------------------------------------------------------------

def embedding_key(p):
    """Hash of exactly the text that gets embedded — same text, same vector."""
    return hashlib.sha1(passage_text(p).encode("utf-8")).hexdigest()[:16]


def previous_vectors(out_dir, model):
    """{embedding key: vector} from the previous build. The index itself is
    the cache: a passage whose embedded text hasn't changed needn't be
    embedded again."""
    import numpy as np
    try:
        info = json.load(open(os.path.join(out_dir, "info.json")))
        meta = [json.loads(line) for line in
                open(os.path.join(out_dir, "meta.jsonl"), encoding="utf-8")]
        vectors = np.load(os.path.join(out_dir, "vectors.npy"))
    except (OSError, ValueError):
        return {}
    if info.get("model") != model or len(meta) != len(vectors):
        return {}       # a different model, or an interrupted build
    # Indexes built before the cache have no "emb"; then the key is recomputed
    # with today's passage_text(), which is only right if it hasn't changed since.
    return {m.get("emb") or embedding_key(m): vectors[i] for i, m in enumerate(meta)}


def build(stub=False, model_name=MODEL_NAME, out_dir=INDEX_DIR,
          bm25_only=False):
    import numpy as np

    records = load_records()
    print(f"{len(records)} passages")
    layers = Counter(p["layer"] for p in records)
    print("  " + "  ".join(f"{k} {v}" for k, v in sorted(layers.items())))
    for p in records:
        p["emb"] = embedding_key(p)

    print("building BM25 …")
    docs = [tokenize(f"{p['kontext']} {p['text']} {p.get('extra_bm25', '')}")
            for p in records]
    bm = BM25(docs)

    vectors = None
    if not bm25_only:
        model = "stub" if stub else model_name
        cache = previous_vectors(out_dir, model)
        missing = {p["emb"]: passage_text(p) for p in records
                   if p["emb"] not in cache}
        print(f"embedding {len(missing)} passages "
              f"({len(records) - sum(p['emb'] in missing for p in records)} "
              f"reused from the previous build) …")
        if missing:
            new = encode(list(missing.values()), stub=stub, model_name=model_name)
            cache.update(zip(missing, new))
        vectors = np.stack([cache[p["emb"]] for p in records]).astype("float32")

    # Everything is written only here, after embedding: an interrupted build
    # leaves the old index untouched instead of half-overwritten.
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "meta.jsonl"), "w", encoding="utf-8") as fh:
        for p in records:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    bm.save(os.path.join(out_dir, "bm25.pkl"))

    if bm25_only:
        print("skipped embedding (--bm25-only); vectors.npy left untouched")
        return records

    np.save(os.path.join(out_dir, "vectors.npy"), vectors)
    with open(os.path.join(out_dir, "info.json"), "w", encoding="utf-8") as fh:
        json.dump({"model": model,
                   "dim": int(vectors.shape[1]), "n": len(records),
                   "max_chars": MAX_CHARS, "overlap": OVERLAP}, fh,
                  ensure_ascii=False, indent=2)

    print(f"\nwrote {out_dir}/  ({vectors.shape[0]} × {vectors.shape[1]})")
    return records


# --------------------------------------------------------------------------
# 5. search
# --------------------------------------------------------------------------

class Index:
    """Holds the index in memory. Use 'shell' for several questions in a row —
    otherwise the embedding model is reloaded on every CLI call."""

    def __init__(self, out_dir=INDEX_DIR):
        import numpy as np
        if not os.path.exists(os.path.join(out_dir, "info.json")):
            sys.exit(f"no index found in {out_dir} — run 'build' first")
        self.info = json.load(open(os.path.join(out_dir, "info.json")))
        self.meta = [json.loads(line) for line in
                     open(os.path.join(out_dir, "meta.jsonl"), encoding="utf-8")]
        self.vectors = np.load(os.path.join(out_dir, "vectors.npy"))
        if len(self.meta) != len(self.vectors):
            sys.exit(f"meta.jsonl ({len(self.meta)} rows) and vectors.npy "
                     f"({len(self.vectors)}) don't match — is a 'build' running, or "
                     f"was --bm25-only run after the chunks changed? Finish a 'build'.")
        self.bm = BM25.load(os.path.join(out_dir, "bm25.pkl"))
        self.stub = self.info["model"] == "stub"

    def _encode_query(self, question):
        """The model is loaded once per Index instance, not per question."""
        if self.stub:
            return stub_vectors([f"query: {question}"])[0]
        if not hasattr(self, "_model"):
            from sentence_transformers import SentenceTransformer
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._model = SentenceTransformer(self.info["model"], device=device)
        return self._model.encode([f"query: {question}"], normalize_embeddings=True,
                                  convert_to_numpy=True).astype("float32")[0]

    def search(self, question, k=8, party=None, layer=None, per_party=None,
               candidates=300, method="hybrid", bm25_weight=None):
        """method: 'hybrid' (BM25 + vector), 'bm25' or 'vector'.

        The two pure modes exist for the ablation in the report — and 'bm25'
        needs no model, which makes it possible to test everything else
        without loading 2 GB.

        Returns [(rrf_score, passage)], best first. When vector search ran,
        each passage also carries "similarity" (cosine similarity to the
        question).
        """
        import numpy as np

        w_bm = RRF_WEIGHT_BM25 if bm25_weight is None else bm25_weight
        sim = None   # vector similarities; set below if the model runs

        # When searching for a specific party, widen the candidate pool to the
        # whole index — otherwise the party's documents can fall outside the
        # top 300 when other parties dominate the topic (e.g. MP+V+S on climate).
        n_candidates = len(self.meta) if party else candidates

        rankings = []
        if method in ("hybrid", "bm25"):
            rankings.append((1.0 if method == "bm25" else w_bm,
                             self.bm.search(question, limit=n_candidates,
                                            drop_party_names=bool(party))))
        if method in ("hybrid", "vector"):
            qv = self._encode_query(question)
            sim = self.vectors @ qv
            rankings.append((RRF_WEIGHT_VECTOR,
                             [(int(i), float(sim[i]))
                              for i in np.argsort(-sim)[:n_candidates]]))

        # Reciprocal Rank Fusion: rankings are merged without having to
        # normalise two completely different score scales against each other.
        rrf = defaultdict(float)
        for weight, ranking in rankings:
            for rank, (i, _) in enumerate(ranking):
                rrf[i] += weight / (RRF_K + rank + 1)

        hits = []
        for i, score in sorted(rrf.items(), key=lambda kv: -kv[1]):
            m = self.meta[i]
            if layer and m["layer"] != layer:
                continue
            if party:
                own = set(p for p in m.get("parti", "").split(";") if p)
                if not own & set(party):
                    continue
            hits.append((score, m, i))

        if per_party:
            count = defaultdict(int)
            capped = []
            for score, m, i in hits:
                keys = [p for p in m.get("parti", "").split(";") if p] or ["—"]
                if all(count[p] >= per_party for p in keys):
                    continue
                for p in keys:
                    count[p] += 1
                capped.append((score, m, i))
            hits = capped

        # The cosine similarity comes along as "similarity" (only when vector
        # search ran): the RRF score is rank-based and says nothing about how
        # CLOSE a hit is to the question, so a relevance cutoff must use it.
        return [(score, m if sim is None else {**m, "similarity": float(sim[i])})
                for score, m, i in hits[:k]]


def show_hits(hits):
    if not hits:
        print("no hits")
        return
    for n, (score, m) in enumerate(hits, 1):
        source = m.get("url") or f"{m.get('beteckning', '')} {m.get('rm', '')}"
        text = m["text"].replace("\n", " ")
        print(f"\n{n:2}. [{score:.4f}] {m['layer'].upper()}  {m['kontext']}")
        print(f"    {text[:260]}{'…' if len(text) > 260 else ''}")
        print(f"    {source}   id={m['id']}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--stub", action="store_true",
                   help="smoke test with a fake encoder, no model is loaded")
    b.add_argument("--model", default=MODEL_NAME)
    b.add_argument("--out", default=INDEX_DIR)
    b.add_argument("--bm25-only", action="store_true",
                   help="rebuild meta + BM25 without touching vectors.npy")

    s = sub.add_parser("search")
    s.add_argument("question")
    s.add_argument("-k", type=int, default=8)
    s.add_argument("--party", default="", help="e.g. V,SD")
    s.add_argument("--layer", choices=["said", "did", "motion"])
    s.add_argument("--per-party", type=int, default=None,
                   help="max number of hits per party")
    s.add_argument("--index", default=INDEX_DIR)
    s.add_argument("--method", choices=["hybrid", "bm25", "vector"], default="hybrid")
    s.add_argument("--bm25-weight", type=float, default=None,
                   help=f"BM25's weight in RRF (default {RRF_WEIGHT_BM25})")

    sh = sub.add_parser("shell", help="several questions in a row, the model loads once")
    sh.add_argument("-k", type=int, default=8)
    sh.add_argument("--index", default=INDEX_DIR)

    a = ap.parse_args()
    if a.cmd == "build":
        build(stub=a.stub, model_name=a.model, out_dir=a.out,
              bm25_only=a.bm25_only)
    elif a.cmd == "shell":
        idx = Index(a.index)
        print(f"{idx.info['n']} passages, model {idx.info['model']}. "
              f"An empty line quits.")
        while True:
            try:
                question = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                break
            show_hits(idx.search(question, k=a.k))
    else:
        idx = Index(a.index)
        parties = [p.strip().upper() for p in a.party.split(",") if p.strip()]
        show_hits(idx.search(a.question, k=a.k, party=parties or None,
                             layer=a.layer, per_party=a.per_party, method=a.method,
                             bm25_weight=a.bm25_weight))


if __name__ == "__main__":
    main()
