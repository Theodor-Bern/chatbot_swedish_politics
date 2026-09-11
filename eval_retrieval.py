"""
eval_retrieval.py — mäter hur bra retrievalen är, innan vi bygger svarssteget.

Facit ligger i eval/questions.json och uttrycks som REGLER över metadata, inte
som handplockade passage-id:n. En träff är relevant om den uppfyller frågans
"relevant"-villkor (rätt lager, rätt parti, rätt ämnesslug, rätt utskott).
Det gör facit granskningsbart av hela gruppen och tåligt mot ombyggda index.

Frågetyperna mäter olika saker:
  smal          hittar vi rätt sida för ETT parti?          -> P@k, MRR
  bred          får ALLA partier plats i kontexten?         -> partitäckning
  saknas        facit är noll träffar                       -> falska träffar
  did           hittar vi rätt ärenden i riksdagsmaterialet -> P@k, MRR
  identifierare hittar vi ett ärende på dess beteckning?    -> P@k, MRR
  jamforelse    är båda lagren representerade?              -> lagertäckning

Frågor utan "relevant"-block (t.ex. typen 'avrader', som testar svarssteget
och inte hämtningen) hoppas över helt.

Kör:
    python eval_retrieval.py                       # hybrid
    python eval_retrieval.py --metod bm25          # kräver ingen modell
    python eval_retrieval.py --metod bm25 --metod vektor --metod hybrid
    python eval_retrieval.py --metod hybrid --vikt-bm25 0.15
    python eval_retrieval.py --index out/index_v2 --detalj B02
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

import rag_index

FRAGOR = "eval/questions.json"
PARTIER = ["S", "M", "SD", "C", "V", "KD", "MP", "L"]


def är_relevant(träff, regel):
    """Uppfyller passagen frågans relevanskriterium?"""
    if not regel:
        return False
    if regel.get("layer") and träff.get("layer") != regel["layer"]:
        return False
    if regel.get("parti"):
        egna = {p for p in (träff.get("parti") or "").split(";") if p}
        if not egna & set(regel["parti"]):
            return False
    if regel.get("beteckning") and träff.get("beteckning") not in regel["beteckning"]:
        return False
    if regel.get("rm") and träff.get("rm") not in regel["rm"]:
        return False
    if regel.get("utskott"):
        # Utskottskoderna är blandade versaler (JuU, SfU, MJU) eftersom de
        # kommer från beteckningens prefix. Jämför skiftlägesokänsligt.
        vill = {u.lower() for u in regel["utskott"]}
        if (träff.get("utskott") or "").lower() not in vill:
            return False
    if regel.get("sakfraga_regex"):
        if not re.search(regel["sakfraga_regex"], träff.get("sakfraga") or "",
                         re.IGNORECASE):
            return False
    if regel.get("text_regex"):
        blob = f"{träff.get('kontext', '')} {träff.get('text', '')}"
        if not re.search(regel["text_regex"], blob, re.IGNORECASE):
            return False
    return True


def kör_fråga(idx, q, metod, vikt_bm25=None):
    sok = q.get("sok", {})
    träffar = idx.search(
        q["fraga"],
        k=sok.get("k", 10),
        parti=sok.get("parti"),
        layer=sok.get("layer"),
        per_parti=sok.get("per_parti"),
        metod=metod,
        vikt_bm25=vikt_bm25,
    )
    metas = [m for _poäng, m in träffar]
    flaggor = [är_relevant(m, q.get("relevant")) for m in metas]

    res = {
        "id": q["id"], "typ": q["typ"], "fraga": q["fraga"],
        "n": len(metas), "n_rel": sum(flaggor),
        "metas": metas, "flaggor": flaggor,
    }
    res["precision"] = (sum(flaggor) / len(flaggor)) if flaggor else 0.0
    res["mrr"] = next((1 / (i + 1) for i, f in enumerate(flaggor) if f), 0.0)

    if q["typ"] == "bred":
        hittade = set()
        for m, f in zip(metas, flaggor):
            if f:
                hittade |= {p for p in (m.get("parti") or "").split(";") if p}
        vänta = set(q.get("forvantade_partier", PARTIER))
        res["partier_hittade"] = sorted(hittade & vänta)
        res["partier_saknade"] = sorted(vänta - hittade)
        res["tackning"] = len(hittade & vänta) / len(vänta) if vänta else 0.0
        # Partier som INTE ska finnas men som ändå fick en "relevant" träff.
        res["oväntade"] = sorted(hittade - vänta)

    if q["typ"] == "saknas":
        res["falska"] = sum(flaggor)
        res["topp"] = metas[0]["kontext"] if metas else "(inga träffar alls)"

    if q["typ"] == "jamforelse":
        lager = {m["layer"] for m, f in zip(metas, flaggor) if f}
        res["lager"] = sorted(lager)
        res["bada_lagren"] = lager == {"said", "did"}

    return res


def sammanfatta(resultat, metod):
    print(f"\n{'=' * 66}\nMETOD: {metod}\n{'=' * 66}")
    per_typ = defaultdict(list)
    for r in resultat:
        per_typ[r["typ"]].append(r)

    print(f"\n{'id':5}{'typ':14}{'P@k':>7}{'MRR':>7}{'täckn':>8}  anmärkning")
    for r in resultat:
        täck = f"{r['tackning']:.2f}" if "tackning" in r else "—"
        anm = ""
        if r["typ"] == "saknas":
            anm = ("OK — inga falska träffar" if r["falska"] == 0
                   else f"{r['falska']} FALSKA: {r['topp'][:44]}")
        elif r["typ"] == "bred" and r.get("partier_saknade"):
            anm = "saknas: " + ",".join(r["partier_saknade"])
        elif r["typ"] == "jamforelse":
            anm = "lager: " + (",".join(r["lager"]) or "inga")
        elif r["n_rel"] == 0:
            anm = "INGEN RELEVANT TRÄFF"
        print(f"{r['id']:5}{r['typ']:14}{r['precision']:7.2f}{r['mrr']:7.2f}"
              f"{täck:>8}  {anm}")

    print("\nper frågetyp:")
    for typ, rs in per_typ.items():          # alla typer som finns i setet
        p = sum(r["precision"] for r in rs) / len(rs)
        mrr = sum(r["mrr"] for r in rs) / len(rs)
        extra = ""
        if typ == "bred":
            extra = f"  partitäckning {sum(r['tackning'] for r in rs) / len(rs):.2f}"
        if typ == "saknas":
            extra = f"  frågor utan falska träffar {sum(1 for r in rs if not r['falska'])}/{len(rs)}"
        if typ == "jamforelse":
            extra = f"  båda lagren {sum(1 for r in rs if r['bada_lagren'])}/{len(rs)}"
        print(f"  {typ:14} n={len(rs):2}  P@k {p:.2f}  MRR {mrr:.2f}{extra}")


def detalj(r):
    print(f"\n--- {r['id']}  {r['fraga']}")
    for i, (m, f) in enumerate(zip(r["metas"], r["flaggor"]), 1):
        print(f"  {i:2} {'REL' if f else '   '} [{m['layer']}] {m['kontext'][:95]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=rag_index.INDEX_DIR)
    ap.add_argument("--fragor", default=FRAGOR)
    ap.add_argument("--metod", action="append",
                    choices=["hybrid", "bm25", "vektor"])
    ap.add_argument("--vikt-bm25", type=float, default=None,
                    help="BM25:s vikt i RRF vid metod=hybrid")
    ap.add_argument("--detalj", action="append", default=[],
                    help="visa alla träffar för dessa fråge-id")
    a = ap.parse_args()

    frågor = [q for q in json.load(open(a.fragor, encoding="utf-8"))
              if q.get("relevant")]        # typer utan facit (t.ex. avrader) hoppas över
    idx = rag_index.Index(a.index)
    print(f"{idx.info['n']} passager, modell {idx.info['model']}, "
          f"{len(frågor)} frågor")

    for metod in (a.metod or ["hybrid"]):
        resultat = [kör_fråga(idx, q, metod, a.vikt_bm25) for q in frågor]
        sammanfatta(resultat, metod)
        for r in resultat:
            if r["id"] in a.detalj:
                detalj(r)


if __name__ == "__main__":
    main()