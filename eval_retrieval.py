"""
eval_retrieval.py — measures how good retrieval is, before the answering step.

Ground truth lives in eval/questions.json and is expressed as RULES over
metadata, not as hand-picked passage ids. A hit is relevant if it meets the
question's "relevant" condition (right layer, right party, right topic slug,
right utskott). That keeps the ground truth reviewable by the whole group and
robust against rebuilt indexes. The file's keys (sok, fraga, typ, relevant, …)
and question types are its data format and stay Swedish.

The question types measure different things:
  smal          do we find the right page for ONE party?       -> P@k, MRR
  bred          do ALL parties get a place in the context?     -> party coverage
  saknas        the ground truth is zero hits                  -> false hits
  did           do we find the right cases in the Riksdag data -> P@k, MRR
  identifierare do we find a case by its beteckning?           -> P@k, MRR
  jamforelse    are both layers represented?                   -> layer coverage

Questions without a "relevant" block (e.g. the type 'avrader', which tests the
answering step, not retrieval) are skipped entirely.

Run:
    python eval_retrieval.py                        # hybrid
    python eval_retrieval.py --method bm25          # needs no model
    python eval_retrieval.py --method bm25 --method vector --method hybrid
    python eval_retrieval.py --method hybrid --bm25-weight 0.15
    python eval_retrieval.py --index out/index_v2 --detail B02
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

import rag_index

QUESTIONS_FILE = "eval/questions.json"
PARTIES = ["S", "M", "SD", "C", "V", "KD", "MP", "L"]


def is_relevant(hit, rule):
    """Does the passage meet the question's relevance criterion?"""
    if not rule:
        return False
    if rule.get("layer") and hit.get("layer") != rule["layer"]:
        return False
    if rule.get("parti"):
        own = {p for p in (hit.get("parti") or "").split(";") if p}
        if not own & set(rule["parti"]):
            return False
    if rule.get("beteckning") and hit.get("beteckning") not in rule["beteckning"]:
        return False
    if rule.get("rm") and hit.get("rm") not in rule["rm"]:
        return False
    if rule.get("utskott"):
        # Utskott codes are mixed case (JuU, SfU, MJU) because they come from
        # the beteckning's prefix. Compare case-insensitively.
        wanted = {u.lower() for u in rule["utskott"]}
        if (hit.get("utskott") or "").lower() not in wanted:
            return False
    if rule.get("sakfraga_regex"):
        if not re.search(rule["sakfraga_regex"], hit.get("sakfraga") or "",
                         re.IGNORECASE):
            return False
    if rule.get("text_regex"):
        blob = f"{hit.get('kontext', '')} {hit.get('text', '')}"
        if not re.search(rule["text_regex"], blob, re.IGNORECASE):
            return False
    return True


def run_question(idx, q, method, bm25_weight=None):
    search = q.get("sok", {})
    hits = idx.search(
        q["fraga"],
        k=search.get("k", 10),
        party=search.get("parti"),
        layer=search.get("layer"),
        per_party=search.get("per_parti"),
        method=method,
        bm25_weight=bm25_weight,
    )
    metas = [m for _score, m in hits]
    flags = [is_relevant(m, q.get("relevant")) for m in metas]

    res = {
        "id": q["id"], "type": q["typ"], "question": q["fraga"],
        "n": len(metas), "n_rel": sum(flags),
        "metas": metas, "flags": flags,
    }
    res["precision"] = (sum(flags) / len(flags)) if flags else 0.0
    res["mrr"] = next((1 / (i + 1) for i, f in enumerate(flags) if f), 0.0)

    if q["typ"] == "bred":
        found = set()
        for m, f in zip(metas, flags):
            if f:
                found |= {p for p in (m.get("parti") or "").split(";") if p}
        expected = set(q.get("forvantade_partier", PARTIES))
        res["parties_found"] = sorted(found & expected)
        res["parties_missing"] = sorted(expected - found)
        res["coverage"] = len(found & expected) / len(expected) if expected else 0.0
        # Parties that should NOT be there but still got a "relevant" hit.
        res["unexpected"] = sorted(found - expected)

    if q["typ"] == "saknas":
        res["false_hits"] = sum(flags)
        res["top"] = metas[0]["kontext"] if metas else "(no hits at all)"

    if q["typ"] == "jamforelse":
        layers = {m["layer"] for m, f in zip(metas, flags) if f}
        res["layers"] = sorted(layers)
        res["both_layers"] = layers == {"said", "did"}

    return res


def summarize(results, method):
    print(f"\n{'=' * 66}\nMETHOD: {method}\n{'=' * 66}")
    per_type = defaultdict(list)
    for r in results:
        per_type[r["type"]].append(r)

    print(f"\n{'id':5}{'type':14}{'P@k':>7}{'MRR':>7}{'cover':>8}  note")
    for r in results:
        cover = f"{r['coverage']:.2f}" if "coverage" in r else "—"
        note = ""
        if r["type"] == "saknas":
            note = ("OK — no false hits" if r["false_hits"] == 0
                    else f"{r['false_hits']} FALSE: {r['top'][:44]}")
        elif r["type"] == "bred" and r.get("parties_missing"):
            note = "missing: " + ",".join(r["parties_missing"])
        elif r["type"] == "jamforelse":
            note = "layers: " + (",".join(r["layers"]) or "none")
        elif r["n_rel"] == 0:
            note = "NO RELEVANT HIT"
        print(f"{r['id']:5}{r['type']:14}{r['precision']:7.2f}{r['mrr']:7.2f}"
              f"{cover:>8}  {note}")

    print("\nper question type:")
    for qtype, rs in per_type.items():          # every type present in the set
        p = sum(r["precision"] for r in rs) / len(rs)
        mrr = sum(r["mrr"] for r in rs) / len(rs)
        extra = ""
        if qtype == "bred":
            extra = f"  party coverage {sum(r['coverage'] for r in rs) / len(rs):.2f}"
        if qtype == "saknas":
            extra = f"  questions without false hits {sum(1 for r in rs if not r['false_hits'])}/{len(rs)}"
        if qtype == "jamforelse":
            extra = f"  both layers {sum(1 for r in rs if r['both_layers'])}/{len(rs)}"
        print(f"  {qtype:14} n={len(rs):2}  P@k {p:.2f}  MRR {mrr:.2f}{extra}")


def show_detail(r):
    print(f"\n--- {r['id']}  {r['question']}")
    for i, (m, f) in enumerate(zip(r["metas"], r["flags"]), 1):
        print(f"  {i:2} {'REL' if f else '   '} [{m['layer']}] {m['kontext'][:95]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=rag_index.INDEX_DIR)
    ap.add_argument("--questions", default=QUESTIONS_FILE)
    ap.add_argument("--method", action="append",
                    choices=["hybrid", "bm25", "vector"])
    ap.add_argument("--bm25-weight", type=float, default=None,
                    help="BM25's weight in RRF when method=hybrid")
    ap.add_argument("--detail", action="append", default=[],
                    help="show every hit for these question ids")
    a = ap.parse_args()

    questions = [q for q in json.load(open(a.questions, encoding="utf-8"))
                 if q.get("relevant")]        # types without ground truth (e.g. avrader) are skipped
    idx = rag_index.Index(a.index)
    print(f"{idx.info['n']} passages, model {idx.info['model']}, "
          f"{len(questions)} questions")

    for method in (a.method or ["hybrid"]):
        results = [run_question(idx, q, method, a.bm25_weight) for q in questions]
        summarize(results, method)
        for r in results:
            if r["id"] in a.detail:
                show_detail(r)


if __name__ == "__main__":
    main()
