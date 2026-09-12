"""Hämta och filtrera motioner från Riksdagens öppna data.

python3 scrapers/motions.py --rm 2023/24 --limit 20 --out out/motions_pilot.jsonl
python3 scrapers/motions.py --out out/motions.jsonl

--limit avser antal granskade motioner per riksmöte, före filtrering.
Råfiler cachas; en avbruten hämtning kan återupptas med samma kommando.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import time
from urllib.parse import urlencode
import requests

ROOT = Path(__file__).resolve().parent.parent
YEARS = ["2022/23", "2023/24", "2024/25", "2025/26"]


def as_list(value):
    return value if isinstance(value, list) else [value] if isinstance(value, dict) else []


def select(document):
    """Returnera (unika undertecknare, urvalsbeslut). Ingen namn-gissning."""
    if "motionen utgår" in document.get("titel", "").lower():
        return [], "utgar"
    people = as_list((document.get("dokintressent") or {}).get("intressent"))
    signers = {}
    for person in people:
        if person.get("roll", "").lower() != "undertecknare":
            continue
        pid = str(person.get("intressent_id") or "").strip()
        if not pid or not person.get("namn") or not person.get("partibet"):
            return [], "ofullstandiga_undertecknare"
        if pid in signers and signers[pid] != person:
            # Samma person får inte kopplas till motstridiga partier/namn.
            old = signers[pid]
            if any(old.get(k) != person.get(k) for k in ("namn", "partibet")):
                return [], "ofullstandiga_undertecknare"
        signers[pid] = person
    result = sorted(signers.values(), key=lambda p: str(p["intressent_id"]))
    return result, "inkludera" if len(result) >= 2 else "en_undertecknare" if result else "saknar_undertecknare"


class MotionText(HTMLParser):
    """Bevara stycken och ordmellanrum, men ta bort HTML/CSS/skript."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        if not self.skip and tag in ("p", "div", "br", "li", "tr", "h1", "h2", "h3"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)
        elif not self.skip and tag in ("p", "div", "li", "tr", "h1", "h2", "h3"):
            self.parts.append("\n")
        elif not self.skip and tag in ("td", "th"):
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def extract_text(raw):
    parser = MotionText()
    parser.feed(raw)
    text = "".join(parser.parts).replace("\xad", "")
    return "\n\n".join(re.sub(r"\s+", " ", line).strip()
                       for line in text.splitlines() if line.strip())


def get_cached(url, path):
    if path.exists():
        return path.read_text(encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            with requests.get(url, headers={"User-Agent": "SwedishPoliticsCourseProject/1.0"}, timeout=45) as response:
                response.raise_for_status()
                raw = response.content.decode("utf-8-sig")
            temp = path.with_suffix(path.suffix + ".tmp")
            temp.write_text(raw, encoding="utf-8")
            temp.replace(path)
            time.sleep(0.25)
            return raw
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def collect(rms, cache, limit=None):
    docs = {}
    for rm in rms:
        count, page = 0, 1
        seen = set()
        expected = None
        while True:
            # Beteckningen är unik inom riksmötet. Standardsortering ger
            # instabila sidgränser när många motioner har samma datum.
            params = urlencode({"doktyp": "mot", "rm": rm, "sz": 200,
                                "sort": "bet", "sortorder": "desc",
                                "p": page, "utformat": "json"})
            raw = get_cached("https://data.riksdagen.se/dokumentlista/?" + params,
                             cache / rm.replace("/", "-") / f"bet-page-{page}.json")
            listing = json.loads(raw)["dokumentlista"]
            expected = int(listing["@traffar"])
            batch = as_list(listing.get("dokument"))
            if not batch and count < expected:
                raise ValueError(f"Tom sida {page} för {rm} trots {expected} träffar")
            for doc in batch:
                if doc.get("rm") != rm or doc.get("doktyp") != "mot":
                    raise ValueError("API:t returnerade fel riksmöte/dokumenttyp")
                if doc["dok_id"] in seen:
                    raise ValueError(f"Upprepat dokument {doc['dok_id']} mellan listsidor: hämtningen avbryts")
                seen.add(doc['dok_id'])
                docs[doc["dok_id"]] = doc
                count += 1
                if limit and count >= limit:
                    break
            print(f"{rm}: läst {count}/{expected} motioner", flush=True)
            if (limit and count >= limit) or not listing.get("@nasta_sida"):
                break
            page += 1
        if not limit and count != expected:
            raise ValueError(f"Ofullständig lista: {count}/{expected} för {rm}")
    return list(docs.values())


def build(rms, cache, out, limit=None):
    documents = collect(rms, cache, limit)
    stats = Counter()
    selected, audit = [], []
    for doc in documents:
        signers, reason = select(doc)
        stats[reason] += 1
        audit.append({"dok_id": doc["dok_id"], "rm": doc["rm"], "beslut": reason,
                      "antal_undertecknare": len(signers)})
        if reason == "inkludera":
            selected.append((doc, signers))
    print(f"Urval: {dict(stats)}", flush=True)
    stamp = datetime.now(timezone.utc).isoformat()

    def prepare(item):
        doc, signers = item
        did = doc["dok_id"]
        if not re.fullmatch(r"[A-Za-z0-9]+", did):
            raise ValueError("Ogiltigt dokument-id")
        url = f"https://data.riksdagen.se/dokument/{did}.html"
        source_path = cache / "html" / f"{did}.html"
        text = extract_text(get_cached(url, source_path))
        if len(text) < 100:
            raise ValueError(f"Tom eller för kort motion: {did}")
        return {"chunk_id": f"motion:{did}", "dok_id": did, "layer": "motion",
                "rm": doc["rm"], "beteckning": doc["beteckning"],
                "datum": doc.get("datum"),
                "hamtad": datetime.fromtimestamp(source_path.stat().st_mtime, timezone.utc).isoformat(),
                "motionstyp": doc.get("subtyp", ""), "heading": doc["titel"],
                "parti": ";".join(sorted({p["partibet"] for p in signers})),
                "undertecknare": [{k: p[k] for k in ("intressent_id", "namn", "partibet")}
                                   for p in signers],
                "antal_undertecknare": len(signers), "url": url, "text": text}

    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(".jsonl.tmp")
    # Två samtidiga hämtningar; cachen gör omkörning möjlig vid avbrott.
    with ThreadPoolExecutor(max_workers=2) as pool, temp.open("w", encoding="utf-8") as fh:
        for n, row in enumerate(pool.map(prepare, selected), 1):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if n % 100 == 0 or n == len(selected):
                print(f"Motionstexter: {n}/{len(selected)}", flush=True)
    temp.replace(out)
    report = {"riksmoten": rms, "limit_per_rm": limit, "granskade": len(documents),
              "urval": dict(stats), "skapad": stamp, "beslut_per_motion": audit,
              "begransning": "Minst två undertecknare är inte ett mått på hela partiets stöd."}
    out.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Skrev {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rm", nargs="+", default=YEARS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/motions")
    parser.add_argument("--out", type=Path, default=ROOT / "out/motions.jsonl")
    args = parser.parse_args()
    if any(not re.fullmatch(r"\d{4}/\d{2}", rm) for rm in args.rm):
        parser.error("Riksmöte ska skrivas t.ex. 2023/24")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit måste vara minst 1")
    build(args.rm, args.cache, args.out, args.limit)


if __name__ == "__main__":
    main()
