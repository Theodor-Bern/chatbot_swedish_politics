import sys
from pathlib import Path

import pandas as pd

from betankande import parse_document
from votes import load_votes

MANIFEST_COLS = ["hangar_id", "dok_id", "rm", "beteckning", "doktyp", "typ",
                 "subtyp", "tempbeteckning", "organ", "mottagare", "nummer",
                 "datum", "systemdatum", "titel", "subtitel", "status",
                 "relaterat_id"]


def load_manifest(path):
    m = pd.read_csv(path, header=None, names=MANIFEST_COLS, dtype=str)
    m["dok_id"] = m["dok_id"].str.upper()
    return m.set_index("dok_id")[["rm", "beteckning", "organ"]].to_dict("index")


def collect_chunks(folder, manifest):
    rows = []
    for path in sorted(Path(folder).rglob("*.html")):
        if not path.is_file():
            continue
        try:
            info, blocks, chunks = parse_document(path)
        except Exception:
            continue
        meta = manifest.get(info["dok_id"], {})
        for c in chunks:
            for punkt in (c.punkter or [None]):
                rows.append({
                    "dok_id": c.dok_id,
                    "rm": meta.get("rm", ""),
                    "beteckning": meta.get("beteckning", ""),
                    "utskott": meta.get("organ", ""),
                    "section": c.section,
                    "number": c.number,
                    "parties": ";".join(c.parties),
                    "punkt": punkt,
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    manifest = load_manifest(sys.argv[1])
    chunks = collect_chunks(sys.argv[2], manifest)
    votes = load_votes(sys.argv[3])

    print(f"\nchunks: {len(chunks)}  with punkt: {chunks['punkt'].notna().sum()}")

    keys = chunks.dropna(subset=["punkt"]).copy()
    keys["punkt"] = keys["punkt"].astype(float)

    vote_keys = votes[["rm", "beteckning", "punkt"]].drop_duplicates()
    vote_keys["found"] = True

    merged = keys.merge(vote_keys, on=["rm", "beteckning", "punkt"], how="left")
    hit = merged["found"].fillna(False)

    print(f"reservations with a punkt: {len(merged)}")
    print(f"  matched to a votering:   {hit.sum()} ({hit.mean():.1%})")
    print(f"  no matching votering:    {(~hit).sum()}")

    print("\nMisses by riksmöte:")
    print(merged[~hit].groupby("rm").size().to_string())
    print("\nExample misses:")
    print(merged[~hit][["dok_id", "rm", "beteckning", "punkt", "parties"]]
          .head(8).to_string(index=False))