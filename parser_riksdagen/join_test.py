"""Check that reservations can be linked to the voting records.

Usage:
    python join_test.py data/html data/voteringar

Two numbers, and only one of them is about our code:

  FORWARD  how many reservations have a votering. This measures parliament,
           not the parser: a punkt is only voted on if someone demands it,
           otherwise it passes by acklamation. Around 40% is normal and no
           amount of work will raise it.

  INVERSE  how many voteringar land on a reservation we extracted. THIS is
           the join test. Anything below ~98% is a real defect worth chasing.
"""

import sys
from pathlib import Path

import pandas as pd

from betankande import (
    RESERVATION, read_html, document_kind, parse_document,
)
from votes import load_votes


def collect_reservations(folder):
    """One row per (reservation, punkt). rm and beteckning come from the
    document itself — never from the filename, whose FöU stems are mojibake,
    and never from the CSV manifest, which is keyed by that filename."""
    rows, skipped, keyless = [], [], 0

    for path in sorted(Path(folder).rglob("*.html")):
        if not path.is_file():
            continue
        raw = read_html(path)
        kind = document_kind(raw)
        if kind != "word":
            skipped.append(kind)
            continue

        info, paragraphs, blocks, chunks = parse_document(path, raw)
        if not info["beteckning"]:
            keyless += 1

        for c in chunks:
            if c.section != RESERVATION:
                continue        # särskilda yttranden are never voted on
            for punkt in (c.punkter or [None]):
                rows.append({
                    "dok_id": c.dok_id,
                    "rm": c.rm,
                    "beteckning": c.beteckning,
                    "number": c.number,
                    "parties": ";".join(c.parties),
                    "punkt": punkt,
                })

    return pd.DataFrame(rows), keyless


def main(html_folder, votes_folder):
    reservations, keyless = collect_reservations(html_folder)
    votes = load_votes(votes_folder)
    vote_keys = votes[["rm", "beteckning", "punkt"]].drop_duplicates()

    print(f"reservations: {len(reservations)}")
    print(f"  documents with no rm/beteckning in the HTML: {keyless}")
    no_punkt = reservations["punkt"].isna().sum()
    print(f"  without a punkt (cannot ever join):          {no_punkt}")

    keyed = reservations.dropna(subset=["punkt"]).copy()
    keyed["punkt"] = keyed["punkt"].astype(float)

    # ---- forward: does this reservation have a votering? -----------------
    marked = vote_keys.copy()
    marked["voted"] = True
    forward = keyed.merge(marked, on=["rm", "beteckning", "punkt"], how="left")
    hit = forward["voted"].notna()
    print(f"\nFORWARD  {len(forward)} reservations with a punkt")
    print(f"  have a votering:   {hit.sum()} ({hit.mean():.1%})")
    print(f"  decided without a recorded vote (acklamation): {(~hit).sum()}")

    # ---- inverse: does this votering belong to a reservation? ------------
    ours = set(zip(reservations["rm"], reservations["beteckning"]))
    in_corpus = vote_keys[[
        (rm, bet) in ours
        for rm, bet in zip(vote_keys["rm"], vote_keys["beteckning"])
    ]].copy()

    res_keys = keyed[["rm", "beteckning", "punkt"]].drop_duplicates()
    res_keys["linked"] = True
    inverse = in_corpus.merge(res_keys, on=["rm", "beteckning", "punkt"],
                              how="left")
    linked = inverse["linked"].notna()
    print(f"\nINVERSE  {len(inverse)} voteringar in documents we parsed")
    print(f"  attached to a reservation: {linked.sum()} ({linked.mean():.1%})")
    print(f"  unattached:                {(~linked).sum()}")

    missed = inverse[~linked]
    if not missed.empty:
        print("\nUnattached voteringar by utskott:")
        prefix = missed["beteckning"].str.extract(r"^([A-ZÅÄÖa-zåäö]+)")[0]
        print(prefix.value_counts().head(10).to_string())
        print("\nExamples:")
        print(missed.head(10).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])