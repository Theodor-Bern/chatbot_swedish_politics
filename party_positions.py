"""
party_positions.py — the DID layer: what the parties actually DID in the Riksdag.

A Riksdag vote always puts the committee's proposal against ONE reservation.
A raw Ja/Nej is therefore unreadable on its own: "SD röstade nej" means nothing
until you know which reservation was on the floor. This file does two things:

  1. resolves which reservation each vote was about (resolve)
  2. states the result in plain Swedish (describe) — wordalisation, so the
     LLM never has to interpret a number itself

The output JSON (out/positions_did.jsonl) keeps Swedish field names
(voteringar, reservationer, partier, …): answer.py and the tests read them.

Run:
    python party_positions.py build      ->  out/positions_did.jsonl
    python party_positions.py report
    python party_positions.py ask 2022/23 AU10 1 C
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

VOTE_DIR = "data/voteringar"
CHUNKS = "out/chunks.jsonl"
OUT = "out/positions_did.jsonl"

# Column order in the Riksdag's vote CSVs (the files have no header).
COL_RM, COL_BET, COL_VOTE_ID, COL_PUNKT = 0, 1, 2, 3
COL_PARTY, COL_VOTE, COL_AVSER, COL_DATE = 6, 8, 9, 13

PARTIES = ["S", "M", "SD", "C", "V", "KD", "MP", "L"]
PARTY_NAMES = {
    "S": "Socialdemokraterna", "M": "Moderaterna", "SD": "Sverigedemokraterna",
    "C": "Centerpartiet", "V": "Vänsterpartiet", "KD": "Kristdemokraterna",
    "MP": "Miljöpartiet", "L": "Liberalerna",
}
LIVE_VOTES = ("Ja", "Nej", "Avstår")
UTSKOTT_RE = re.compile(r"^([A-Za-zÅÄÖåäö]+)")

# How similar the set of parties must be before we dare to guess.
JACCARD_MIN = 0.34

csv.field_size_limit(10_000_000)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_votes(folder=VOTE_DIR):
    """The CSV files -> {votering_id: {..., 'counts': {party: Counter(vote)}}}.

    We key on votering_id, not on punkt: five punkter in the material have two
    separate votes (two reservations tried one at a time), and they must not
    be merged.
    """
    votes = {}
    paths = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not paths:
        sys.exit(f"no CSV files found in {folder}")
    for path in paths:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.reader(fh):
                if len(row) <= COL_DATE:
                    continue
                vote_id = row[COL_VOTE_ID]
                v = votes.get(vote_id)
                if v is None:
                    v = votes[vote_id] = {
                        "votering_id": vote_id,
                        "rm": row[COL_RM],
                        "beteckning": row[COL_BET],
                        "punkt": row[COL_PUNKT].strip(),
                        "avser": row[COL_AVSER],
                        "datum": row[COL_DATE],
                        "counts": defaultdict(Counter),
                    }
                v["counts"][row[COL_PARTY]][row[COL_VOTE]] += 1
    return votes


def load_reservations(chunks_path=CHUNKS):
    """chunks.jsonl -> reservations, known punkter, titles and riksmöten.

        res[(rm, bet, punkt)]  = [{'number': n, 'parties': frozenset}, ...]
        punkter                = every (rm, bet, punkt) we know of
        corpus_rm              = riksmöten actually present in the corpus
    """
    res = defaultdict(list)
    punkter = set()
    titles = {}
    seen = set()
    if not os.path.exists(chunks_path):
        sys.exit(f"{chunks_path} not found — run build_corpus.py first")

    with open(chunks_path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            rm = (d.get("rm") or "").strip()
            bet = (d.get("beteckning") or "").strip()
            punkt = str(d.get("punkter") or "").strip()
            if not rm or not bet:
                continue
            if d.get("doc_title"):
                titles.setdefault((rm, bet), d["doc_title"])
            if not punkt:
                continue
            key = (rm, bet, punkt)
            punkter.add(key)
            if d.get("section") != "reservation":
                continue
            num = d.get("number")
            if num is None or (key, num) in seen:
                continue
            seen.add((key, num))
            parties = tuple(p for p in str(d.get("parties") or "").split(";") if p)
            res[key].append({"number": int(num), "parties": frozenset(parties)})

    for lst in res.values():
        lst.sort(key=lambda r: r["number"])
    corpus_rm = {rm for rm, _bet in titles}
    return res, punkter, titles, corpus_rm


# --------------------------------------------------------------------------
# resolution: which reservation was the vote about?
# --------------------------------------------------------------------------

def stance_of(counter):
    """The party's stance = the vote most of its members cast.

    Absence doesn't count as a stance; a party that was entirely absent gets
    'Frånvarande'.
    """
    live = {k: v for k, v in counter.items() if k in LIVE_VOTES and v}
    if not live:
        return "Frånvarande"
    return max(live.items(), key=lambda kv: (kv[1], -LIVE_VOTES.index(kv[0])))[0]


def no_bloc(vote):
    """The parties that voted for the reservation (i.e. Nej to the committee)."""
    return frozenset(
        p for p, c in vote["counts"].items()
        if p in PARTIES and stance_of(c) == "Nej"
    )


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def resolve(bloc, candidates, taken, exact_only=False):
    """Match the Nej bloc against the reservations' sets of parties."""
    free = [c for c in candidates if c["number"] not in taken]
    if bloc:
        for c in free:
            if c["parties"] == bloc:
                return c, "exact"
    if exact_only:
        return None, "unknown"
    best, score = None, 0.0
    for c in free:
        s = jaccard(bloc, c["parties"])
        if s > score:
            best, score = c, s
    if best is not None and score >= JACCARD_MIN:
        return best, "partial"
    return None, "unknown"


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def build(vote_dir=VOTE_DIR, chunks_path=CHUNKS, out_path=OUT):
    votes = load_votes(vote_dir)
    res, punkter, titles, corpus_rm = load_reservations(chunks_path)

    by_key = defaultdict(list)
    for v in votes.values():
        by_key[(v["rm"], v["beteckning"], v["punkt"])].append(v)

    stats = Counter()
    rows = []
    all_keys = sorted(punkter | set(by_key))

    for key in all_keys:
        rm, bet, punkt = key
        cands = res.get(key, [])
        vs = sorted(by_key.get(key, []), key=lambda v: (v["avser"], v["votering_id"]))

        issue_votes = [v for v in vs if v["avser"] == "sakfrågan"]
        taken = set()
        assigned = {}
        # Two passes: exact matches first, so a partial guess can't claim a
        # reservation that belongs to another vote on the same punkt.
        for pass_exact in (True, False):
            for v in issue_votes:
                if v["votering_id"] in assigned:
                    continue
                c, how = resolve(no_bloc(v), cands, taken, exact_only=pass_exact)
                if c is None and pass_exact:
                    continue
                assigned[v["votering_id"]] = (c, how)
                if c:
                    taken.add(c["number"])

        out_votes = []
        for v in vs:
            if v["rm"] not in corpus_rm:
                c, how = None, "utanfor_korpus"
            elif v["avser"] == "sakfrågan":
                c, how = assigned.get(v["votering_id"], (None, "unknown"))
            else:
                c, how = None, "motivfraga"
            stats[how] += 1
            out_votes.append({
                "votering_id": v["votering_id"],
                "datum": v["datum"],
                "avser": v["avser"],
                "reservation": c["number"] if c else None,
                "reservation_partier": sorted(c["parties"]) if c else [],
                "resolution": how,
                "nej_bloc": sorted(no_bloc(v)),
                "partier": {
                    p: {
                        "stance": stance_of(v["counts"][p]),
                        "ja": v["counts"][p].get("Ja", 0),
                        "nej": v["counts"][p].get("Nej", 0),
                        "avstod": v["counts"][p].get("Avstår", 0),
                        "franvarande": v["counts"][p].get("Frånvarande", 0),
                    }
                    for p in PARTIES if p in v["counts"]
                },
            })

        m = UTSKOTT_RE.match(bet)
        rows.append({
            "key": f"{rm}|{bet}|{punkt}",
            "rm": rm,
            "beteckning": bet,
            "punkt": punkt,
            "utskott": m.group(1).upper() if m else "",
            "doc_title": titles.get((rm, bet), ""),
            "status": "voted" if out_votes else "no_vote",
            "reservationer": [
                {"number": c["number"], "partier": sorted(c["parties"])} for c in cands
            ],
            "voteringar": out_votes,
        })
        stats["punkter"] += 1
        stats["punkter_" + rows[-1]["status"]] += 1

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    return rows, stats


# --------------------------------------------------------------------------
# wordalisation (Swedish sentences: they go into Gemini's prompt)
# --------------------------------------------------------------------------

def _reservation_label(v):
    """How the vote is referred to."""
    if v["reservation"] is None:
        return "en reservation som inte kunnat identifieras", []
    parties = ", ".join(v["reservation_partier"])
    txt = f"reservation {v['reservation']} ({parties})"
    if v["resolution"] == "partial":
        txt += " [trolig matchning]"
    return txt, v["reservation_partier"]


def _sentence(party, v, info):
    """One sentence about what the party did in a single vote."""
    label, res_parties = _reservation_label(v)
    own = party in res_parties
    stance = info["stance"]

    if stance == "Nej" and own:
        others = [p for p in res_parties if p != party]
        s = f"{party} röstade för sin egen reservation {v['reservation']}"
        if others:
            s += f", som partiet står bakom tillsammans med {', '.join(others)}"
        if v["resolution"] == "partial":
            s += " [trolig matchning]"
        s += "."
    elif stance == "Nej":
        s = f"{party} röstade för {label}."
    elif stance == "Ja" and own:
        s = (f"{party} röstade med utskottets majoritet mot {label} — "
             f"trots att partiet självt står bakom den.")
    elif stance == "Ja":
        s = f"{party} röstade med utskottets majoritet mot {label}."
    elif stance == "Avstår":
        s = f"{party} avstod i omröstningen om {label}."
    else:
        s = f"{party} var frånvarande i omröstningen om {label}."

    cast = [(info["ja"], "ja"), (info["nej"], "nej"), (info["avstod"], "avstod")]
    if sum(1 for n, _ in cast if n) > 1:
        parts = ", ".join(f"{n} {word}" for n, word in cast if n)
        s += f" Partiet var splittrat: {parts}."
    return s


def describe(row, party):
    """Plain Swedish on what a party did on a punkt.

    Returns {'status', 'text', 'own_reservations'}. "Ingen votering hölls" is
    a full answer, not an error — about two thirds of the punkter are decided
    by acclamation.
    """
    name = PARTY_NAMES.get(party, party)
    where = f"{row['beteckning']} ({row['rm']}) punkt {row['punkt']}"
    own = [r["number"] for r in row["reservationer"] if party in r["partier"]]

    if row["status"] == "no_vote":
        text = (f"Ingen votering hölls på {where}; ärendet avgjordes med acklamation. "
                f"{name} tog därför inte ställning i en omröstning.")
        if own:
            text += (f" Partiet hade dock reservation "
                     f"{', '.join(str(n) for n in own)} på punkten.")
        return {"status": "no_vote", "text": text, "own_reservations": own}

    sentences = []
    voted_on = set()
    for v in row["voteringar"]:
        info = v["partier"].get(party)
        if info is None:
            continue
        if v["resolution"] == "utanfor_korpus":
            sentences.append(
                f"{party} deltog i en omröstning på {where} den {v['datum']}, "
                f"men betänkandet för {row['rm']} ingår inte i materialet, "
                f"så vilken reservation omröstningen gällde är okänt."
            )
            continue
        if v["avser"] == "motivfrågan":
            sentences.append(
                f"{party} {'röstade för' if info['stance'] == 'Nej' else 'röstade mot'} "
                f"en motivreservation på {where} (omröstningen gällde motiveringen, "
                f"inte sakfrågan)."
            )
            continue
        if v["reservation"] is not None:
            voted_on.add(v["reservation"])
        sentences.append(_sentence(party, v, info))

    if not sentences:
        return {"status": "no_vote",
                "text": f"{name} deltog inte i någon omröstning på {where}.",
                "own_reservations": own}

    untouched = [n for n in own if n not in voted_on]
    if untouched:
        sentences.append(
            f"{party} hade en egen reservation "
            f"({', '.join(str(n) for n in untouched)}) på punkten, "
            f"som inte var uppe i denna omröstning."
        )
    return {"status": "voted", "text": " ".join(sentences), "own_reservations": own}


# --------------------------------------------------------------------------
# lookup
# --------------------------------------------------------------------------

def load_positions(path=OUT):
    if not os.path.exists(path):
        sys.exit(f"{path} not found — run 'python {sys.argv[0]} build' first")
    with open(path, encoding="utf-8") as fh:
        return {(d := json.loads(line))["key"]: d for line in fh}


def ask(positions, rm, beteckning, punkt, party):
    row = positions.get(f"{rm}|{beteckning}|{punkt}")
    if row is None:
        return {"status": "unknown_punkt",
                "text": f"Punkt {punkt} i {beteckning} ({rm}) finns inte i materialet."}
    return describe(row, party)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def cmd_build():
    rows, stats = build()
    tot = stats["exact"] + stats["partial"] + stats["unknown"]
    print(f"wrote {OUT}  ({len(rows)} punkter)")
    print(f"  with a vote     {stats['punkter_voted']}")
    print(f"  acclamation     {stats['punkter_no_vote']}")
    print(f"\nissue votes within the corpus: {tot}")
    for k in ("exact", "partial", "unknown"):
        n = stats[k]
        print(f"  {k:9} {n:6}  {n / tot * 100:5.1f}%" if tot else f"  {k}: {n}")
    print(f"reasoning (motivfråga) votes: {stats['motivfraga']}")
    if stats["utanfor_korpus"]:
        print(f"\nvotes outside the corpus: {stats['utanfor_korpus']}"
              f"  (riksmöten with no betänkanden in the material)")


def cmd_report():
    positions = load_positions()
    per_party = Counter()
    per_utskott = Counter()
    for row in positions.values():
        for v in row["voteringar"]:
            if v["avser"] != "sakfrågan":
                continue
            per_utskott[row["utskott"]] += 1
            for p, info in v["partier"].items():
                if info["stance"] in LIVE_VOTES:
                    per_party[p] += 1
    print("votes where the party took a stance:")
    for p in PARTIES:
        print(f"  {p:3} {per_party[p]:6}")
    print("\nvotes per utskott (top 12):")
    for u, n in per_utskott.most_common(12):
        print(f"  {u:5} {n:5}")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd = sys.argv[1]
    if cmd == "build":
        cmd_build()
    elif cmd == "report":
        cmd_report()
    elif cmd == "ask":
        if len(sys.argv) != 6:
            sys.exit("ask <rm> <beteckning> <punkt> <party>   "
                     "e.g. ask 2022/23 AU10 1 C")
        positions = load_positions()
        print(ask(positions, *sys.argv[2:6])["text"])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
