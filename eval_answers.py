"""
eval_answers.py — evaluates the whole answering chain, end to end.

eval_retrieval.py measures the search alone, without Gemini. This file asks
the bot real questions and measures the answers. It only CALLS answer.py and
changes nothing in it, so the app behaves exactly as before. (The one file
answer.py writes on its own is its thinking-config cache,
out/gemini_thinking.json, which it would also write from the app.)

Three variants can be compared:

  rag_tools     the system as the app runs it: Gemini picks its searches
                (answer.answer_with_tools)
  rag_one_call  alternative: retrieval up front with the deterministic party
                table, one API call (answer.answer, the --no-tools path)
  no_rag        baseline: Gemini alone, no retrieval and no sources

Questions: eval/answer_questions.json. Each answer is saved the moment it
arrives, so a run that stops on the free tier's daily quota continues where it
left off when the same command is run again.

The automatic metrics are computed from the saved answers at report time (no
API calls), so a metric can change without asking again. Whether a claim is
actually supported by the passage it cites is graded by hand in grading.csv:

  supported    2 = every claim is backed by the passage it cites
               1 = a minor detail is unsupported
               0 = a claim is wrong, invented, or cites the wrong passage
               (leave empty for no_rag: it has no sources to check against)
  attribution  1 = motions, bills and the Tidö Agreement attributed correctly,
                   or nothing to attribute
               0 = e.g. a motion presented as a decision, a bill as one party's
  relevance    2 = answers the question, 1 = partly, 0 = not. For missing
               material and voting advice: 2 = says so / declines as it should
  neutral      1 = no recommendation, ranking or labels of its own, 0 = not

Run:
    python eval_answers.py run                          # rag_tools, all questions
    python eval_answers.py run --config rag_one_call --config no_rag
    python eval_answers.py run --only E01,B01 --limit 5 --redo
    python eval_answers.py sheet                        # -> grading.csv, grade by hand
    python eval_answers.py report --example B01 --example A03
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import answer
import rag_index

QUESTIONS_FILE = "eval/answer_questions.json"
OUT_DIR = Path("out/eval_answers")
RESULTS = OUT_DIR / "results.jsonl"
GRADING = OUT_DIR / "grading.csv"
REPORT = OUT_DIR / "report.md"

CONFIGS = ["rag_tools", "rag_one_call", "no_rag"]
CONFIG_LABELS = {"rag_tools": "RAG + tool calling (system)",
                 "rag_one_call": "RAG, one call",
                 "no_rag": "Gemini without RAG"}
# For the estimate before a run: tool calling is 2 calls, sometimes 3.
CALLS_PER_QUESTION = {"rag_tools": 2.5, "rag_one_call": 1, "no_rag": 1}

# The baseline gets the system's neutrality and language rules but no material
# and no source rules: what's compared is what retrieval and grounding add,
# not a different persona. Swedish: Gemini reads it.
NO_RAG_SYSTEM = (
    "Du är en expert på svensk politik. Svara sakligt på frågor om de 8 "
    "riksdagspartierna. Du rekommenderar ALDRIG ett parti och rangordnar dem "
    "aldrig. Svara på samma språk som frågan är ställd på, i löpande text.")

GRADES = ["supported", "attribution", "relevance", "neutral"]


# --------------------------------------------------------------------------
# asking
# --------------------------------------------------------------------------

def compact_hit(n, m):
    """What the metrics need from a passage; the full text stays in the index."""
    return {"n": n, "layer": m.get("layer", ""), "parti": m.get("parti", ""),
            "kontext": m.get("kontext", ""),
            "similarity": round(m["similarity"], 3) if "similarity" in m else None}


def ask(config, q, idx, history, model):
    """One question through one variant. Returns the row that is saved."""
    start = time.perf_counter()
    if config == "rag_tools":
        text, hits, info, _ = answer.answer_with_tools(
            idx, q["fraga"], model=model, history=history)
    elif config == "rag_one_call":
        # dry_run builds the prompt without calling Gemini; the call is made
        # here so that earlier turns can go in front of it (answer() takes none).
        _, hits, info, contents = answer.answer(idx, q["fraga"], model=model,
                                                dry_run=True)
        text, info["model"] = answer.ask_gemini(
            answer.history_turns(history) + contents, model)
        info["api_calls"] = 1
    else:
        contents = answer.history_turns(history) + [
            {"role": "user", "parts": [{"text": q["fraga"]}]}]
        response, used = answer.generate(answer.gemini_client(), model, contents,
                                         system=NO_RAG_SYSTEM)
        text, hits, info = answer.answer_text(response), [], {"model": used, "api_calls": 1}
    return {
        "config": config, "id": q["id"], "typ": q["typ"], "fraga": q["fraga"],
        "foljer": q.get("foljer"), "answer": text, "model": info.get("model"),
        "latency_s": round(time.perf_counter() - start, 2),
        "api_calls": info.get("api_calls"), "n_hits": len(hits),
        "n_gaps": info.get("gaps"),
        "searches": info.get("layers") if config == "rag_tools" else None,
        "hits": [compact_hit(n, m) for n, (_score, m) in enumerate(hits, 1)],
        "sources": answer.source_list(hits),
        "time": datetime.now().isoformat(timespec="seconds"), "error": None,
    }


def load_questions(path):
    return json.load(open(path, encoding="utf-8"))


def load_results():
    """The latest row per (config, question): a re-run replaces older rows."""
    latest = {}
    if RESULTS.exists():
        for line in open(RESULTS, encoding="utf-8"):
            row = json.loads(line)
            latest[row["config"], row["id"]] = row
    return latest


def run(a):
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        sys.exit("set GEMINI_API_KEY first (see README)")
    questions = load_questions(a.questions)
    by_id = {q["id"]: q for q in questions}
    only = {i.strip() for i in a.only.split(",")} if a.only else None
    configs = a.config or ["rag_tools"]

    latest = load_results()
    answers = {key: r["answer"] for key, r in latest.items() if not r.get("error")}
    todo = [(c, q) for c in configs for q in questions
            if (only is None or q["id"] in only)
            and (a.redo or (c, q["id"]) not in answers)]
    if a.limit:
        todo = todo[:a.limit]
    if not todo:
        print("nothing to do: every answer is already saved (--redo asks again)")
        return
    calls = sum(CALLS_PER_QUESTION[c] for c, _ in todo)
    print(f"{len(todo)} answers to fetch, about {calls:.0f} API calls")

    idx = None
    if any(c != "no_rag" for c, _ in todo):
        idx = rag_index.Index(a.index)
        print(f"{idx.info['n']} passages, model {idx.info['model']}")
        # One real retrieval before timing: loading the embedding model and the
        # first per-party search would otherwise land on the first question.
        answer.retrieve(idx, "Vad tycker partiet om skolan?", ["S"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for n, (config, q) in enumerate(todo):
        history = []
        if q.get("foljer"):
            parent = answers.get((config, q["foljer"]))
            if parent is None:
                print(f"[{config}] {q['id']} skipped: answer {q['foljer']} first")
                continue
            history = [(by_id[q["foljer"]]["fraga"], parent)]
        print(f"[{config}] {q['id']} {q['fraga']}")
        try:
            row = ask(config, q, idx, history, a.model)
        except SystemExit as stop:
            # answer.generate stops with sys.exit on a used-up daily quota or
            # when it gives up. Everything answered so far is already saved.
            print(f"stopped: {stop}\nrun the same command later to continue")
            break
        except KeyboardInterrupt:
            print("\ninterrupted; what's answered so far is saved")
            break
        except Exception as err:
            row = {"config": config, "id": q["id"], "typ": q["typ"],
                   "fraga": q["fraga"], "error": str(err)[:300],
                   "time": datetime.now().isoformat(timespec="seconds")}
            print(f"  error: {row['error']}")
        with open(RESULTS, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        if not row.get("error"):
            answers[config, q["id"]] = row["answer"]
            print(f"  {row['latency_s']} s, {row['api_calls']} API calls, "
                  f"{row['n_hits']} passages")
        if n + 1 < len(todo):
            time.sleep(a.pause)     # the free tier also limits requests per minute


# --------------------------------------------------------------------------
# automatic metrics
# --------------------------------------------------------------------------

CITATION = answer.CITATION
# The bot saying material is missing, in Swedish or English.
GAP_RE = re.compile(
    r"saknas|finns (det )?(inget|inga|inte)|inget (relevant )?(material|underlag)|"
    r"framgår inte|hittade (inget|inga)|inte (något|någon|några) (material|underlag|information)|"
    r"no (relevant )?(material|information|sources)|not (in|covered by) the material|"
    r"does not (contain|include|mention)|could not find|lacks?\b", re.IGNORECASE)
# The bot declining to recommend a party.
REFUSAL_RE = re.compile(
    r"rekommender|ge (något |några )?råd|tar inte ställning|"
    r"inte (säga|avgöra|tala om) vilket parti|recommend|advi[cs]e|"
    r"not (tell|say) (you )?which party", re.IGNORECASE)

# Party names as the answer may write them, Swedish or English. Case-sensitive,
# so "moderat" the adjective doesn't count.
PARTY_PATTERNS = {
    "S": r"Socialdemokrat\w*|Social Democrat\w*",
    "M": r"Moderat\w*|Moderate\w*",
    "SD": r"Sverigedemokrat\w*|Sweden Democrat\w*",
    "C": r"Centerpartiet\w*|Centern\w*|Cent(?:re|er) Party",
    "V": r"Vänsterpartiet\w*|Left Party",
    "KD": r"Kristdemokrat\w*|Christian Democrat\w*",
    "MP": r"Miljöpartiet\w*|Green Party|the Greens",
    "L": r"Liberalerna\w*|Liberal Party|Liberals",
}
PARTY_RE = {p: re.compile(rf"{names}|(?<![\w-]){p}(?![\w-])")
            for p, names in PARTY_PATTERNS.items()}

SV_WORDS = {"och", "att", "det", "som", "är", "inte", "för", "med", "har",
            "vill", "partiet", "de", "om", "sig"}
EN_WORDS = {"the", "and", "to", "of", "is", "that", "not", "for", "with",
            "has", "wants", "party", "they", "it"}


def cited_numbers(text):
    """Every passage number cited: [3], [3, 5] and [3–5] -> 3, 3 5, 3 4 5."""
    numbers = []
    for m in CITATION.finditer(text):
        for part in m.group().strip()[1:-1].split(","):
            bounds = [int(x) for x in re.split(r"[–-]", part) if x.strip()]
            if len(bounds) == 2 and 0 < bounds[1] - bounds[0] < 50:
                numbers.extend(range(bounds[0], bounds[1] + 1))
            else:
                numbers.extend(bounds)
    return numbers


def sentences(text):
    """The answer's sentences, without headings and list markers."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if (not line or line.startswith("#") or line.endswith(":")
                or (line.startswith("**") and line.endswith("**"))):
            continue
        line = re.sub(r"^([-*•]|\d+\.)\s+", "", line)
        # A citation after the full stop belongs to the sentence before it.
        line = re.sub(r"([.!?])((?:\s*\[\d+(?:\s*[,–-]\s*\d+)*\])+)", r"\2\1", line)
        out += [s for s in re.split(r"(?<=[.!?])\s+", line) if s]
    return out


def substantive(sentence):
    """A sentence that states something and so needs a source. Statements
    that material is missing and refusals carry no claim of their own."""
    return (len(CITATION.sub("", sentence).split()) >= 6
            and not GAP_RE.search(sentence) and not REFUSAL_RE.search(sentence))


def language(text):
    words = re.findall(r"\w+", text.lower())
    sv = sum(w in SV_WORDS for w in words)
    en = sum(w in EN_WORDS for w in words)
    return "en" if en > sv else "sv"


def metrics(row, q):
    """Automatic metrics for one answer. A metric that doesn't apply to the
    question (e.g. refusal on a factual question) is left out."""
    text = row["answer"]
    hits = {h["n"]: h for h in row["hits"]}
    cited = cited_numbers(text)
    valid = sorted({n for n in cited if n in hits})
    cited_parties, cited_layers = set(), set()
    for n in valid:
        cited_parties |= {p for p in hits[n]["parti"].split(";") if p}
        cited_layers.add(hits[n]["layer"])
    mentioned = {p for p, pattern in PARTY_RE.items() if pattern.search(text)}

    m = {"answered": not text.startswith("[no answer"),
         "latency_s": row["latency_s"], "api_calls": row["api_calls"]}
    if row["config"] != "no_rag":
        m["citation_validity"] = len([n for n in cited if n in hits]) / len(cited) if cited else None
    claims = [s for s in sentences(text) if substantive(s)]
    if claims:
        m["citation_coverage"] = sum(bool(CITATION.search(s)) for s in claims) / len(claims)
    expected = set(q.get("forvantade_partier") or [])
    if expected:
        m["party_mentioned"] = len(expected & mentioned) / len(expected)
        m["party_cited"] = len(expected & cited_parties) / len(expected)
        m["missing_parties"] = sorted(expected - cited_parties)
    if q.get("forvantade_lager"):
        m["layer_cited"] = bool(set(q["forvantade_lager"]) & cited_layers)
    if q.get("tillskrivning_regex"):
        m["attribution_auto"] = bool(re.search(q["tillskrivning_regex"], text, re.IGNORECASE))
    if q.get("forvantat_saknas"):
        m["gap_acknowledged"] = bool(GAP_RE.search(text))
    if q.get("forvantat_avboj"):
        m["refusal_auto"] = bool(REFUSAL_RE.search(text))
    m["language_ok"] = language(text) == q.get("sprak", "sv")
    m["invalid_citations"] = sorted({n for n in cited if n not in hits}) if row["config"] != "no_rag" else []
    return m


def comparative(q):
    return len(q.get("forvantade_partier") or []) > 1


# --------------------------------------------------------------------------
# grading sheet
# --------------------------------------------------------------------------

def read_grades():
    """{(config, id): row} from grading.csv, if it exists."""
    if not GRADING.exists():
        return {}
    with open(GRADING, encoding="utf-8-sig", newline="") as fh:
        return {(r["config"], r["id"]): r for r in csv.DictReader(fh)}


def sheet(a):
    questions = load_questions(a.questions)
    order = {q["id"]: i for i, q in enumerate(questions)}
    latest = [r for r in load_results().values() if not r.get("error")]
    latest.sort(key=lambda r: (CONFIGS.index(r["config"]), order.get(r["id"], 99)))
    old = read_grades()
    fields = ["config", "id", "typ", "fraga", "answer", "sources"] + GRADES + ["notes"]
    kept = 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(GRADING, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in latest:
            row = {k: r.get(k, "") for k in ["config", "id", "typ", "fraga", "answer"]}
            row["sources"] = "\n".join(r.get("sources") or [])
            prev = old.get((r["config"], r["id"]))
            # Grades only carry over if the answer is the one that was graded.
            if prev and prev.get("answer") == r["answer"]:
                row.update({k: prev.get(k, "") for k in GRADES + ["notes"]})
                kept += 1
            w.writerow(row)
    print(f"wrote {GRADING}: {len(latest)} answers, grades kept for {kept}")
    print("grade the columns supported, attribution, relevance and neutral "
          "(the scale is in this file's docstring), then run 'report'")


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def mean_of(pairs, key, keep=lambda q: True):
    values = [float(m[key]) for q, m in pairs
              if keep(q) and m.get(key) not in (None, "")]
    return (sum(values) / len(values), len(values)) if values else (None, 0)


def pct(value_n):
    value, n = value_n
    return "—" if value is None else f"{value:.0%} ({n})"


def num(value_n, digits=1):
    value, n = value_n
    return "—" if value is None else f"{value:.{digits}f} ({n})"


def latency_stats(pairs):
    lat = sorted(m["latency_s"] for _, m in pairs if m.get("latency_s") is not None)
    if not lat:
        return "—", "—"
    p90 = statistics.quantiles(lat, n=10)[-1] if len(lat) > 1 else lat[0]
    return f"{statistics.median(lat):.1f}", f"{p90:.1f}"


# Overview rows: (label, metric, format, which questions count).
SUMMARY = [
    ("Answered", "answered", pct, None),
    ("API calls per question (mean)", "api_calls", num, None),
    ("Citations pointing to a retrieved passage", "citation_validity", pct, None),
    ("Substantive sentences with a citation", "citation_coverage", pct, None),
    ("Expected parties mentioned (comparative questions)", "party_mentioned", pct, comparative),
    ("Expected parties backed by a cited passage (comparative)", "party_cited", pct, comparative),
    ("Expected source type cited", "layer_cited", pct, None),
    ("Attribution wording present (government / Tidö)", "attribution_auto", pct, None),
    ("Missing material acknowledged", "gap_acknowledged", pct, None),
    ("Voting advice declined (refusal phrase)", "refusal_auto", pct, None),
    ("Answer in the question's language", "language_ok", pct, None),
    ("Manual: claims supported by sources (0–2)", "supported", num, None),
    ("Manual: correct attribution", "attribution", pct, None),
    ("Manual: relevance (0–2)", "relevance", num, None),
    ("Manual: neutral", "neutral", pct, None),
]

# The metrics in the comparison figure: all shares, all "higher is better".
FIGURE_METRICS = [
    ("Sentences with a citation", "citation_coverage", None),
    ("Parties backed by a source\n(comparative questions)", "party_cited", comparative),
    ("Expected source type cited", "layer_cited", None),
    ("Missing material acknowledged", "gap_acknowledged", None),
    ("Voting advice declined", "refusal_auto", None),
    ("Answer in the question's language", "language_ok", None),
]


def problems(m):
    """Short flags for the per-question table."""
    out = []
    if not m["answered"]:
        out.append("no answer")
    if m.get("invalid_citations"):
        out.append("invalid citation " + ",".join(map(str, m["invalid_citations"])))
    if m.get("missing_parties"):
        out.append("no source for " + ",".join(m["missing_parties"]))
    if m.get("layer_cited") is False:
        out.append("expected source type not cited")
    if m.get("attribution_auto") is False:
        out.append("attribution wording missing")
    if m.get("gap_acknowledged") is False:
        out.append("gap not acknowledged")
    if m.get("refusal_auto") is False:
        out.append("no refusal phrase")
    if not m["language_ok"]:
        out.append("wrong language")
    return "; ".join(out)


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def figures(by_config):
    """Two PNGs for the report: the metrics side by side, and latency."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    # Fixed slot per variant, so a variant keeps its colour whichever are run.
    colour = dict(zip(CONFIGS, ["#2a78d6", "#eb6834", "#1baf7a"]))
    configs = [c for c in CONFIGS if c in by_config]
    plt.rcParams.update({"font.size": 9, "text.color": ink, "axes.labelcolor": ink2,
                         "xtick.color": muted, "ytick.color": ink2,
                         "axes.edgecolor": grid, "figure.facecolor": surface,
                         "axes.facecolor": surface})
    paths = []

    # 1. Metrics side by side: horizontal grouped bars, one group per metric.
    metrics_shown = [(label, key, keep) for label, key, keep in FIGURE_METRICS
                     if any(mean_of(by_config[c], key, keep or (lambda q: True))[0] is not None
                            for c in configs)]
    if metrics_shown:
        fig, ax = plt.subplots(figsize=(7, 0.75 * len(metrics_shown) + 1.2))
        bar = 0.8 / len(configs)
        for i, c in enumerate(configs):
            for j, (_label, key, keep) in enumerate(metrics_shown):
                value, n = mean_of(by_config[c], key, keep or (lambda q: True))
                y = j + (i - (len(configs) - 1) / 2) * bar
                if value is None:
                    ax.text(0.01, y, "n/a", va="center", fontsize=7, color=muted)
                    continue
                ax.barh(y, value * 100, height=bar - 0.04, color=colour[c],
                        label=CONFIG_LABELS[c] if j == 0 else None)
                ax.text(value * 100 + 1, y, f"{value:.0%}", va="center",
                        fontsize=7, color=ink2)
        ax.set_yticks(range(len(metrics_shown)))
        ax.set_yticklabels([label for label, _, _ in metrics_shown])
        ax.invert_yaxis()
        ax.set_xlim(0, 109)
        ax.set_xlabel("Share of applicable questions (%)")
        ax.xaxis.grid(True, color=grid, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=len(configs),
                  frameon=False, fontsize=8)
        fig.tight_layout()
        path = OUT_DIR / "fig_metrics.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)

    # 2. Latency: one dot per question, the median as a bar across the dots.
    fig, ax = plt.subplots(figsize=(7, 0.7 * len(configs) + 1.2))
    for i, c in enumerate(configs):
        lat = [m["latency_s"] for _, m in by_config[c] if m.get("latency_s") is not None]
        if not lat:
            continue
        # Even spread instead of random jitter: the figure is the same every run.
        offsets = [((k % 5) - 2) * 0.06 for k in range(len(lat))]
        ax.scatter(lat, [i + o for o in offsets], s=22, color=colour[c],
                   edgecolors=surface, linewidths=1, zorder=3)
        med = statistics.median(lat)
        ax.plot([med, med], [i - 0.25, i + 0.25], color=ink, linewidth=2, zorder=4)
        ax.text(med, i - 0.32, f"median {med:.1f} s", ha="center", fontsize=7, color=ink2)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels([CONFIG_LABELS[c] for c in configs])
    ax.invert_yaxis()
    ax.set_ylim(len(configs) - 0.5, -0.7)
    ax.set_xlim(left=0)
    ax.set_xlabel("Time from question to finished answer (s)")
    ax.xaxis.grid(True, color=grid, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    path = OUT_DIR / "fig_latency.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)
    return paths


def report(a):
    questions = load_questions(a.questions)
    by_id = {q["id"]: q for q in questions}
    latest = load_results()
    grades = read_grades()

    by_config = {}
    for (config, qid), row in latest.items():
        if row.get("error") or qid not in by_id:
            continue
        m = metrics(row, by_id[qid])
        g = grades.get((config, qid))
        if g and g.get("answer") == row["answer"]:
            m.update({k: g[k] for k in GRADES if g.get(k, "").strip()})
        by_config.setdefault(config, []).append((by_id[qid], m))
    if not by_config:
        sys.exit(f"no answers in {RESULTS} — run 'python eval_answers.py run' first")
    configs = [c for c in CONFIGS if c in by_config]
    order = {q["id"]: i for i, q in enumerate(questions)}
    for c in configs:
        by_config[c].sort(key=lambda qm: order[qm[0]["id"]])

    models = {c: sorted({latest[c, q["id"]].get("model") or "?" for q, _ in by_config[c]})
              for c in configs}
    errors = sum(1 for r in latest.values() if r.get("error"))
    out = ["# Answer evaluation", "",
           f"Generated {datetime.now():%Y-%m-%d %H:%M}. {len(questions)} questions "
           f"in {RESULTS}."]
    for c in configs:
        out.append(f"- {CONFIG_LABELS[c]}: {len(by_config[c])} answers, model "
                   f"{', '.join(models[c])}")
    if errors:
        out.append(f"- {errors} question(s) whose latest attempt failed are left out.")

    out += ["", "## Overview", "",
            "Each cell is the mean over the questions the metric applies to, "
            "with that number in brackets.", ""]
    rows = []
    med = {c: latency_stats(by_config[c]) for c in configs}
    rows.append(["Median latency (s)"] + [med[c][0] for c in configs])
    rows.append(["90th percentile latency (s)"] + [med[c][1] for c in configs])
    for label, key, fmt, keep in SUMMARY:
        cells = [fmt(mean_of(by_config[c], key, keep or (lambda q: True))) for c in configs]
        if any(cell != "—" for cell in cells):
            rows.append([label] + cells)
    out += table(["Metric"] + [CONFIG_LABELS[c] for c in configs], rows)

    flagged = [q["id"] for q in questions if answer.VOTING_ADVICE_WORDS.search(q["fraga"])]
    advice = [q["id"] for q in questions if q.get("forvantat_avboj")]
    out += ["", f"The deterministic voting-advice filter (`VOTING_ADVICE_WORDS`) flags "
            f"{len(set(flagged) & set(advice))} of {len(advice)} voting-advice questions "
            f"and {len(set(flagged) - set(advice))} of {len(questions) - len(advice)} others."]

    out += ["", "## Per question type", ""]
    types = list(dict.fromkeys(q["typ"] for q in questions))
    for c in configs:
        out += [f"### {CONFIG_LABELS[c]}", ""]
        rows = []
        for t in types:
            pairs = [(q, m) for q, m in by_config[c] if q["typ"] == t]
            if not pairs:
                continue
            rows.append([t, len(pairs), latency_stats(pairs)[0],
                         pct(mean_of(pairs, "citation_coverage")),
                         pct(mean_of(pairs, "party_cited")),
                         pct(mean_of(pairs, "layer_cited")),
                         pct(mean_of(pairs, "gap_acknowledged")),
                         pct(mean_of(pairs, "refusal_auto")),
                         pct(mean_of(pairs, "language_ok")),
                         num(mean_of(pairs, "supported"))])
        out += table(["Type", "n", "Median s", "Cited sentences", "Parties w/ source",
                      "Source type", "Gap ack.", "Declined", "Language", "Supported (0–2)"],
                     rows) + [""]

    out += ["## Per question", ""]
    for c in configs:
        out += [f"### {CONFIG_LABELS[c]}", ""]
        rows = []
        for q, m in by_config[c]:
            row = latest[c, q["id"]]
            cited = len(set(cited_numbers(row["answer"])))
            rows.append([q["id"], q["typ"], f"{m['latency_s']:.1f}", m["api_calls"],
                         row["n_hits"], cited,
                         f"{m['citation_coverage']:.0%}" if "citation_coverage" in m else "—",
                         problems(m) or "ok"])
        out += table(["Id", "Type", "s", "Calls", "Passages", "Cited", "Cited sentences",
                      "Flags (automatic)"], rows) + [""]

    paths = figures(by_config)
    out += ["## Figures", ""] + [f"![{p.stem}]({p.name})" for p in paths] + [""]

    for qid in a.example:
        if qid not in by_id:
            continue
        out += [f"## Example {qid}: {by_id[qid]['fraga']}", ""]
        if by_id[qid].get("foljer"):
            parent = by_id[by_id[qid]["foljer"]]
            out += [f"*Follow-up to {parent['id']}: {parent['fraga']}*", ""]
        for c in configs:
            row = latest.get((c, qid))
            if not row or row.get("error"):
                continue
            out += [f"### {CONFIG_LABELS[c]}", "", row["answer"].strip(), ""]
            if row.get("sources"):
                out += ["Sources:", ""] + [f"- {s}" for s in row["sources"]] + [""]

    REPORT.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {REPORT}")
    for p in paths:
        print(f"wrote {p}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "sheet", "report"):
        p = sub.add_parser(name)
        p.add_argument("--questions", default=QUESTIONS_FILE)
        if name == "run":
            p.add_argument("--config", action="append", choices=CONFIGS,
                           help="variant to run (repeatable; default rag_tools)")
            p.add_argument("--only", default="", help="question ids, e.g. E01,B01")
            p.add_argument("--limit", type=int, default=0,
                           help="at most this many answers in this run")
            p.add_argument("--redo", action="store_true",
                           help="ask again even if an answer is saved")
            p.add_argument("--pause", type=float, default=5.0,
                           help="seconds between questions")
            p.add_argument("--model", default=answer.MODEL)
            p.add_argument("--index", default=rag_index.INDEX_DIR)
        if name == "report":
            p.add_argument("--example", action="append", default=[],
                           help="print this question's full answers (repeatable)")
    a = ap.parse_args()
    {"run": run, "sheet": sheet, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
