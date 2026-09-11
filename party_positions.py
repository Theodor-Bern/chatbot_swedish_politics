"""
party_positions.py — DID-lagret: vad partierna faktiskt GJORDE i riksdagen.

En votering i riksdagen ställer alltid utskottets förslag mot EN reservation.
Rå Ja/Nej är därför oläsbart i sig: "SD röstade nej" betyder ingenting förrän
man vet vilken reservation som var uppe. Den här filen gör två saker:

  1. löser upp vilken reservation varje votering gällde (resolve)
  2. formulerar resultatet i klartext (describe) — wordalisation, så att
     LLM:en aldrig behöver tolka en siffra själv

Kör:
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

# Kolumnordningen i riksdagens votering-CSV (ingen header i filerna).
C_RM, C_BET, C_VID, C_PUNKT = 0, 1, 2, 3
C_PARTI, C_ROST, C_AVSER, C_DATUM = 6, 8, 9, 13

PARTIES = ["S", "M", "SD", "C", "V", "KD", "MP", "L"]
PARTY_NAMES = {
    "S": "Socialdemokraterna", "M": "Moderaterna", "SD": "Sverigedemokraterna",
    "C": "Centerpartiet", "V": "Vänsterpartiet", "KD": "Kristdemokraterna",
    "MP": "Miljöpartiet", "L": "Liberalerna",
}
LIVE_VOTES = ("Ja", "Nej", "Avstår")
UTSKOTT_RE = re.compile(r"^([A-Za-zÅÄÖåäö]+)")

# Hur lik partiuppsättningen måste vara för att vi ska våga gissa.
JACCARD_MIN = 0.34

csv.field_size_limit(10_000_000)


# --------------------------------------------------------------------------
# inläsning
# --------------------------------------------------------------------------

def load_votes(folder=VOTE_DIR):
    """CSV-filerna -> {votering_id: {..., 'counts': {parti: Counter(rost)}}}.

    Vi nycklar på votering_id, inte på punkt: fem punkter i materialet har
    två separata voteringar (två reservationer prövade var för sig), och de
    ska inte slås ihop.
    """
    voteringar = {}
    paths = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not paths:
        sys.exit(f"hittar inga CSV-filer i {folder}")
    for path in paths:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.reader(fh):
                if len(row) <= C_DATUM:
                    continue
                vid = row[C_VID]
                v = voteringar.get(vid)
                if v is None:
                    v = voteringar[vid] = {
                        "votering_id": vid,
                        "rm": row[C_RM],
                        "beteckning": row[C_BET],
                        "punkt": row[C_PUNKT].strip(),
                        "avser": row[C_AVSER],
                        "datum": row[C_DATUM],
                        "counts": defaultdict(Counter),
                    }
                v["counts"][row[C_PARTI]][row[C_ROST]] += 1
    return voteringar


def load_reservations(chunks_path=CHUNKS):
    """chunks.jsonl -> reservationer, kända punkter, titlar och riksmöten.

        res[(rm, bet, punkt)]  = [{'number': n, 'parties': frozenset}, ...]
        punkter                = alla (rm, bet, punkt) vi känner till
        corpus_rm              = riksmöten som faktiskt finns i korpusen
    """
    res = defaultdict(list)
    punkter = set()
    titles = {}
    seen = set()
    if not os.path.exists(chunks_path):
        sys.exit(f"hittar inte {chunks_path} — kör build_corpus.py först")

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
# upplösning: vilken reservation gällde omröstningen?
# --------------------------------------------------------------------------

def stance_of(counter):
    """Partiets hållning = den röst flest av dess ledamöter lade.

    Frånvaro räknas inte som en hållning; ett parti som var helt frånvarande
    får 'Frånvarande'.
    """
    live = {k: v for k, v in counter.items() if k in LIVE_VOTES and v}
    if not live:
        return "Frånvarande"
    return max(live.items(), key=lambda kv: (kv[1], -LIVE_VOTES.index(kv[0])))[0]


def nej_bloc(votering):
    """Partierna som röstade för reservationen (dvs nej till utskottet)."""
    return frozenset(
        p for p, c in votering["counts"].items()
        if p in PARTIES and stance_of(c) == "Nej"
    )


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def resolve(bloc, candidates, taken, exact_only=False):
    """Matcha nej-blocket mot reservationernas partiuppsättningar."""
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
    voteringar = load_votes(vote_dir)
    res, punkter, titles, corpus_rm = load_reservations(chunks_path)

    by_key = defaultdict(list)
    for v in voteringar.values():
        by_key[(v["rm"], v["beteckning"], v["punkt"])].append(v)

    stats = Counter()
    rows = []
    all_keys = sorted(punkter | set(by_key))

    for key in all_keys:
        rm, bet, punkt = key
        cands = res.get(key, [])
        vs = sorted(by_key.get(key, []), key=lambda v: (v["avser"], v["votering_id"]))

        sak = [v for v in vs if v["avser"] == "sakfrågan"]
        taken = set()
        assigned = {}
        # Två pass: exakta matchningar först, så att en partiell gissning
        # inte kan lägga beslag på en reservation som hör till en annan
        # omröstning på samma punkt.
        for pass_exact in (True, False):
            for v in sak:
                if v["votering_id"] in assigned:
                    continue
                c, how = resolve(nej_bloc(v), cands, taken, exact_only=pass_exact)
                if c is None and pass_exact:
                    continue
                assigned[v["votering_id"]] = (c, how)
                if c:
                    taken.add(c["number"])

        out_voteringar = []
        for v in vs:
            if v["rm"] not in corpus_rm:
                c, how = None, "utanfor_korpus"
            elif v["avser"] == "sakfrågan":
                c, how = assigned.get(v["votering_id"], (None, "unknown"))
            else:
                c, how = None, "motivfraga"
            stats[how] += 1
            out_voteringar.append({
                "votering_id": v["votering_id"],
                "datum": v["datum"],
                "avser": v["avser"],
                "reservation": c["number"] if c else None,
                "reservation_partier": sorted(c["parties"]) if c else [],
                "resolution": how,
                "nej_bloc": sorted(nej_bloc(v)),
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
            "status": "voted" if out_voteringar else "no_vote",
            "reservationer": [
                {"number": c["number"], "partier": sorted(c["parties"])} for c in cands
            ],
            "voteringar": out_voteringar,
        })
        stats["punkter"] += 1
        stats["punkter_" + rows[-1]["status"]] += 1

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    return rows, stats


# --------------------------------------------------------------------------
# wordalisation
# --------------------------------------------------------------------------

def _mot(v):
    """Hur omröstningen ska benämnas."""
    if v["reservation"] is None:
        return "en reservation som inte kunnat identifieras", []
    part = ", ".join(v["reservation_partier"])
    txt = f"reservation {v['reservation']} ({part})"
    if v["resolution"] == "partial":
        txt += " [trolig matchning]"
    return txt, v["reservation_partier"]


def _mening(parti, v, info):
    """En mening om vad partiet gjorde i en enskild omröstning."""
    mot, res_partier = _mot(v)
    egen = parti in res_partier
    stance = info["stance"]

    if stance == "Nej" and egen:
        andra = [p for p in res_partier if p != parti]
        s = f"{parti} röstade för sin egen reservation {v['reservation']}"
        if andra:
            s += f", som partiet står bakom tillsammans med {', '.join(andra)}"
        if v["resolution"] == "partial":
            s += " [trolig matchning]"
        s += "."
    elif stance == "Nej":
        s = f"{parti} röstade för {mot}."
    elif stance == "Ja" and egen:
        s = (f"{parti} röstade med utskottets majoritet mot {mot} — "
             f"trots att partiet självt står bakom den.")
    elif stance == "Ja":
        s = f"{parti} röstade med utskottets majoritet mot {mot}."
    elif stance == "Avstår":
        s = f"{parti} avstod i omröstningen om {mot}."
    else:
        s = f"{parti} var frånvarande i omröstningen om {mot}."

    avlagda = [(info["ja"], "ja"), (info["nej"], "nej"), (info["avstod"], "avstod")]
    if sum(1 for n, _ in avlagda if n) > 1:
        delar = ", ".join(f"{n} {ord_}" for n, ord_ in avlagda if n)
        s += f" Partiet var splittrat: {delar}."
    return s


def describe(row, parti):
    """Klartext om vad ett parti gjorde på en punkt.

    Returnerar {'status', 'text', 'egna_reservationer'}. "Ingen votering hölls"
    är ett fullvärdigt svar, inte ett fel — omkring två tredjedelar av
    punkterna avgörs med acklamation.
    """
    namn = PARTY_NAMES.get(parti, parti)
    var = f"{row['beteckning']} ({row['rm']}) punkt {row['punkt']}"
    egna = [r["number"] for r in row["reservationer"] if parti in r["partier"]]

    if row["status"] == "no_vote":
        text = (f"Ingen votering hölls på {var}; ärendet avgjordes med acklamation. "
                f"{namn} tog därför inte ställning i en omröstning.")
        if egna:
            text += (f" Partiet hade dock reservation "
                     f"{', '.join(str(n) for n in egna)} på punkten.")
        return {"status": "no_vote", "text": text, "egna_reservationer": egna}

    meningar = []
    prövade = set()
    for v in row["voteringar"]:
        info = v["partier"].get(parti)
        if info is None:
            continue
        if v["resolution"] == "utanfor_korpus":
            meningar.append(
                f"{parti} deltog i en omröstning på {var} den {v['datum']}, "
                f"men betänkandet för {row['rm']} ingår inte i materialet, "
                f"så vilken reservation omröstningen gällde är okänt."
            )
            continue
        if v["avser"] == "motivfrågan":
            meningar.append(
                f"{parti} {'röstade för' if info['stance'] == 'Nej' else 'röstade mot'} "
                f"en motivreservation på {var} (omröstningen gällde motiveringen, "
                f"inte sakfrågan)."
            )
            continue
        if v["reservation"] is not None:
            prövade.add(v["reservation"])
        meningar.append(_mening(parti, v, info))

    if not meningar:
        return {"status": "no_vote",
                "text": f"{namn} deltog inte i någon omröstning på {var}.",
                "egna_reservationer": egna}

    orörda = [n for n in egna if n not in prövade]
    if orörda:
        meningar.append(
            f"{parti} hade en egen reservation "
            f"({', '.join(str(n) for n in orörda)}) på punkten, "
            f"som inte var uppe i denna omröstning."
        )
    return {"status": "voted", "text": " ".join(meningar), "egna_reservationer": egna}


# --------------------------------------------------------------------------
# lookup
# --------------------------------------------------------------------------

def load_positions(path=OUT):
    if not os.path.exists(path):
        sys.exit(f"hittar inte {path} — kör 'python {sys.argv[0]} build' först")
    return {json.loads(l)["key"]: json.loads(l) for l in open(path, encoding="utf-8")}


def ask(positions, rm, beteckning, punkt, parti):
    row = positions.get(f"{rm}|{beteckning}|{punkt}")
    if row is None:
        return {"status": "unknown_punkt",
                "text": f"Punkt {punkt} i {beteckning} ({rm}) finns inte i materialet."}
    return describe(row, parti)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def cmd_build():
    rows, stats = build()
    tot = stats["exact"] + stats["partial"] + stats["unknown"]
    print(f"skrev {OUT}  ({len(rows)} punkter)")
    print(f"  med votering   {stats['punkter_voted']}")
    print(f"  acklamation    {stats['punkter_no_vote']}")
    print(f"\nsakfrågevoteringar inom korpusen: {tot}")
    for k in ("exact", "partial", "unknown"):
        n = stats[k]
        print(f"  {k:9} {n:6}  {n / tot * 100:5.1f}%" if tot else f"  {k}: {n}")
    print(f"motivfrågevoteringar: {stats['motivfraga']}")
    if stats["utanfor_korpus"]:
        print(f"\nvoteringar utanför korpusen: {stats['utanfor_korpus']}"
              f"  (riksmöten det saknas betänkanden för)")


def cmd_report():
    positions = load_positions()
    per_parti = Counter()
    per_utskott = Counter()
    for row in positions.values():
        for v in row["voteringar"]:
            if v["avser"] != "sakfrågan":
                continue
            per_utskott[row["utskott"]] += 1
            for p, info in v["partier"].items():
                if info["stance"] in LIVE_VOTES:
                    per_parti[p] += 1
    print("voteringar där partiet tog ställning:")
    for p in PARTIES:
        print(f"  {p:3} {per_parti[p]:6}")
    print("\nvoteringar per utskott (topp 12):")
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
            sys.exit("ask <rm> <beteckning> <punkt> <parti>   "
                     "t.ex. ask 2022/23 AU10 1 C")
        positions = load_positions()
        print(ask(positions, *sys.argv[2:6])["text"])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()